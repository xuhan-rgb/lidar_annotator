"""KITTI entry points, camera isolation and full-pose pipeline checks."""
import json,os,tempfile,time,unittest
from threading import Event
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from PySide6.QtWidgets import QApplication
from .cli import REPO
from .storage import sample_path,default_source
from .fresh_pipeline import run,prepare,poses
from .qt_app import AnnotationWindow,annotate_command
from .reset_results import clear_previous_results
from .result_runs import latest_result,remember_result


def sequence(root):
    seq=root/'sequence';(seq/'velodyne').mkdir(parents=True)
    for camera,color in [('image_2','red'),('image_3','blue')]:
        (seq/camera).mkdir()
        for frame in range(2):Image.new('RGB',(20,10),color).save(seq/camera/f'{frame:06d}.png')
    for frame in range(2):np.array([[0,0,5,1]],dtype='<f4').tofile(seq/'velodyne'/f'{frame:06d}.bin')
    (seq/'calib.txt').write_text('P2: 10 0 10 0 0 10 5 0 0 0 1 0\nP3: 10 0 10 -2 0 10 5 0 0 0 1 0\nTr: 1 0 0 0 0 1 0 0 0 0 1 0\n')
    (seq/'times.txt').write_text('0\n0.1\n')
    (seq/'poses.txt').write_text('1 0 0 0 0 1 0 0 0 0 1 0\n1 0 0 0.1 0 1 0 0 0 0 1 0\n')
    return seq


class KittiWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])

    def wait_for_frame(self,window,frame):
        deadline=time.monotonic()+3
        while time.monotonic()<deadline:
            self.app.processEvents()
            if window.frame_reader is None and window.camera_view.data is not None and window.camera_view.data['frame']==frame:return
            time.sleep(.01)
        self.fail(window.preview_status.text())

    def test_projection_navigation_reads_each_selected_frame_before_processing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);seq=sequence(root);Image.new('RGB',(20,10),'blue').save(seq/'image_2/000001.png')
            window=AnnotationWindow(sample_path('testdata/zgc_review/profile.json'))
            try:
                window.input_format.setCurrentIndex(window.input_format.findData('kitti'))
                window.lmdb_root.setText(str(seq));window.source_changed();window.lmdb_output.setText(str(root/'work'))
                window.projection_button.click();self.wait_for_frame(window,0)
                self.assertEqual(window.camera_view.data['image'].pixelColor(0,0).getRgb()[:3],(255,0,0))
                window.next_frame.click();self.wait_for_frame(window,1)
                self.assertEqual(window.camera_view.data['image'].pixelColor(0,0).getRgb()[:3],(0,0,255))
                self.assertEqual(window.bev_view.data['frame'],1)
                window.frame_slider.setValue(0);self.wait_for_frame(window,0)
                self.assertEqual(window.preview_frame.value(),0)
                other=sequence(root/'other');Image.new('RGB',(20,10),'green').save(other/'image_2/000000.png')
                window.lmdb_root.setText(str(other));window.source_changed();window.projection_button.click();self.wait_for_frame(window,0)
                self.assertEqual(window.camera_view.data['image'].pixelColor(0,0).getRgb()[:3],(0,128,0))
            finally:window.close()

    def test_projection_switch_while_loading_finishes_on_latest_frame_without_timer(self):
        from .live_preview import load_frame as actual_load
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);seq=sequence(root);Image.new('RGB',(20,10),'blue').save(seq/'image_2/000001.png')
            window=AnnotationWindow(sample_path('testdata/zgc_review/profile.json'));started=Event();release=Event()
            def delayed(context,frame,*args,**kwargs):
                if frame==0 and not started.is_set():started.set();release.wait(3)
                return actual_load(context,frame,*args,**kwargs)
            try:
                window.preview_timer.stop()
                window.input_format.setCurrentIndex(window.input_format.findData('kitti'));window.lmdb_root.setText(str(seq));window.source_changed();window.lmdb_output.setText(str(root/'work'))
                with patch('tools.lidar_annotator.live_preview.load_frame',side_effect=delayed):
                    window.projection_button.click();self.assertTrue(started.wait(2))
                    window.next_frame.click();self.assertEqual(window.preview_frame.value(),1);release.set()
                    self.wait_for_frame(window,1)
                    self.assertEqual(window.camera_view.data['image'].pixelColor(0,0).getRgb()[:3],(0,0,255))
                    self.assertEqual(window.bev_view.data['frame'],1)
            finally:release.set();window.close()

    def test_kitti_source_uses_relative_display_and_absolute_read_path(self):
        window=AnnotationWindow(sample_path('testdata/zgc_review/profile.json'))
        try:
            window.input_format.setCurrentIndex(window.input_format.findData('kitti'))
            self.assertTrue((REPO/window.lmdb_root.text()).is_dir())
            with tempfile.TemporaryDirectory() as tmp:
                seq=sequence(Path(tmp))
                with patch('tools.lidar_annotator.qt_app.QFileDialog.getExistingDirectory',return_value=str(seq)):
                    window.browse_lmdb()
                self.assertEqual(window.lmdb_root.text(),os.path.relpath(seq,REPO))
                self.assertEqual(window.camera_channel.currentData(),'image_2')
                args=annotate_command(window.lmdb_root.text(),'/tmp/job',['render'],1,input_format='kitti',camera='image_2')
                self.assertEqual(Path(args[args.index('--source')+1]).resolve(),seq)
                window.lmdb_root.setText(str(seq));window.source_changed()
                self.assertEqual(window.lmdb_root.text(),os.path.relpath(seq,REPO))
        finally:window.close()

    def test_format_help_opens_in_app_with_templates_and_example_folder(self):
        window=AnnotationWindow(sample_path('testdata/zgc_review/profile.json'))
        try:
            self.assertFalse(hasattr(window,'format_help_button'))
            window.result_tabs.setCurrentWidget(window.help_page)
            self.assertIs(window.result_tabs.currentWidget(),window.help_page)
            text=window.format_help.toPlainText()
            for expected in ('后续算法使用内部缓存','不需要先转换成 LMDB','导出 KITTI 只是数据格式转换','image_2/','velodyne/','P2: 800 0 640','Tr: 0 -1 0','0.100000000','Tr_velo_to_cam_3','小端 float32','处理与结果查看'):
                self.assertIn(expected,text)
            minimal=sample_path('examples/kitti_minimal_two_frames')
            self.assertIn(str(minimal),text)
            self.assertIn('两帧最小示例目录（点击打开）',text)
            from PySide6.QtCore import QUrl
            with patch('tools.lidar_annotator.qt_app.QDesktopServices.openUrl') as open_url:
                window.format_help.anchorClicked.emit(QUrl.fromLocalFile(str(minimal)))
                self.assertEqual(Path(open_url.call_args.args[0].toLocalFile()),minimal)
            with patch('tools.lidar_annotator.qt_app.QDesktopServices.openUrl') as open_url:
                window.open_format_example.click()
                self.assertEqual(Path(open_url.call_args.args[0].toLocalFile()),default_source('kitti'))
        finally:window.close()

    def test_help_toc_contains_headings_and_navigates_to_their_blocks(self):
        from PySide6.QtCore import Qt
        window=AnnotationWindow(sample_path('testdata/zgc_review/profile.json'))
        try:
            self.assertIs(window.help_splitter.widget(0),window.help_toc)
            self.assertIs(window.help_splitter.widget(1),window.format_help)
            def walk(item):
                yield item
                for i in range(item.childCount()):yield from walk(item.child(i))
            items=[item for i in range(window.help_toc.topLevelItemCount()) for item in walk(window.help_toc.topLevelItem(i))]
            titles=[item.text(0) for item in items]
            for title in ('LMDB 输入：先解析，再导出 KITTI 与执行标注','KITTI 输入：直接读取已有序列','最小结构','标定和位姿约定','项目内可直接读取的两帧最小示例','处理与结果查看'):
                self.assertIn(title,titles)
            self.assertNotIn('每行依次是 x、y、z（米）和 intensity',titles)
            for item in items:
                window.help_toc.itemClicked.emit(item,0)
                self.assertEqual(window.format_help.textCursor().position(),item.data(0,Qt.ItemDataRole.UserRole))
                self.assertEqual(window.format_help.textCursor().block().text(),item.text(0))
        finally:window.close()

    def test_camera_selection_changes_output_and_preserves_lmdb_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            seq=sequence(Path(tmp));window=AnnotationWindow(sample_path('testdata/zgc_review/profile.json'))
            try:
                with patch.object(window,'refresh_preview'),patch.object(window,'refresh_test_history'):
                    window.input_format.setCurrentIndex(window.input_format.findData('lmdb'));lmdb=window.lmdb_root.text()
                    window.input_format.setCurrentIndex(window.input_format.findData('kitti'))
                    window.lmdb_root.setText(str(seq));window.source_changed();window.frames.setText('0-1')
                    self.assertEqual([window.camera_channel.itemData(i) for i in range(window.camera_channel.count())],['image_2','image_3'])
                    a=window.base_context();self.assertEqual(a['camera'],'image_2')
                    window.camera_channel.setCurrentIndex(1);b=window.base_context()
                    self.assertEqual(b['camera'],'image_3');self.assertNotEqual(a['output'],b['output'])
                    self.assertEqual(window.recipe_button.text(),'编辑类别与提示词…')
                    args=annotate_command(str(seq),b['output'],['render'],1,input_format=b['input_format'],camera=b['camera'])
                    self.assertEqual(args[args.index('--camera')+1],'image_3')
                    window.input_format.setCurrentIndex(window.input_format.findData('lmdb'));self.assertEqual(window.lmdb_root.text(),lmdb)
            finally:window.close()

    def test_camera_identity_blocks_reuse_and_reset_of_other_camera(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);seq=sequence(root);job=root/'job'
            plan=run(seq,job,stages=['render'],camera='image_2',input_format='kitti',dry_run=True)
            self.assertEqual(plan['job']['camera'],'image_2');self.assertFalse(job.exists())
            job.mkdir();(job/'job.json').write_text(json.dumps(plan['job']))
            with self.assertRaisesRegex(ValueError,'different source'):run(seq,job,stages=['render'],camera='image_3',dry_run=True)
            a=dict(output=str(job),review_output=str(job/'review'),input_format='kitti',camera='image_2')
            b={**a,'camera':'image_3'}
            self.assertEqual(clear_previous_results(b,b,seq,None,dry_run=True),[])
            self.assertEqual(clear_previous_results(a,a,seq,None,dry_run=True),[job])
            out=job.parent/(job.name+'_runs')/'run';result={**a,'output':str(out),'review_output':str(out/'review')}
            remember_result(a,seq,None,result)
            self.assertEqual(latest_result(a,seq,None),result)
            self.assertEqual(latest_result(b,seq,None),b)

    def test_supplied_full_rotation_survives_pose_preparation(self):
        with tempfile.TemporaryDirectory() as tmp:
            job=Path(tmp);(job/'raw').mkdir();prepare(job)
            transforms=np.repeat(np.eye(4)[None],2,axis=0);angle=.04
            transforms[1,:3,:3]=[[1,0,0],[0,np.cos(angle),-np.sin(angle)],[0,np.sin(angle),np.cos(angle)]]
            transforms[1,:3,3]=[.1,.02,.03]
            np.savez(job/'raw/odometry.npz',poses=np.zeros((2,4)),transforms=transforms,timestamps=np.array([0,100000000],np.int64))
            poses(job)
            with np.load(job/'temporal/poses.npz') as z:np.testing.assert_allclose(z['transforms'],transforms)
            np.savez(job/'raw/odometry.npz',poses=np.zeros((2,4)),transforms=transforms,timestamps=np.array([np.nan,np.nan]))
            with self.assertRaisesRegex(ValueError,'时间戳'):poses(job)
            transforms[:]=np.nan
            np.savez(job/'raw/odometry.npz',poses=np.zeros((2,4)),transforms=transforms,timestamps=np.array([0,100000000],np.int64))
            with self.assertRaisesRegex(ValueError,'位姿'):poses(job)

if __name__=='__main__':unittest.main()
