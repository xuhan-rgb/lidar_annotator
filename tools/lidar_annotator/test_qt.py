"""Offscreen component checks; no desktop window is shown or captured."""
import json,os,time,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from PySide6.QtCore import QProcess,QUrl
from PySide6.QtWidgets import QApplication
from .cli import DEFAULT,REPO
from .storage import sample_path
from .qt_app import ICON_PATH,AnnotationWindow,annotate_command,command_for


def use_cached_work(window,out):
    """Component fixtures represent an already selected historical workspace."""
    base=window.base_context();work=dict(output=str(out),review_output=str(Path(out)/'review'))
    window.result_contexts[window.result_context_key(base)]={**base,**work}


class QtWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])

    def test_workflow_and_output_are_text_without_visible_checkboxes(self):
        window=AnnotationWindow(DEFAULT)
        try:
            for widget in [*window.checks.values(),window.temporal_flow,window.export_flow]:
                self.assertTrue(widget.isHidden())
            for text in ('语义标注与目标跟踪（SAM3）','行人 / 车辆实例标签（YOLO26x、WildDet3D；复用 SAM mask）','行人 / 车辆时序优化','生成标签与可视化'):
                from PySide6.QtWidgets import QLabel
                self.assertTrue(any(label.text()==text and not label.isHidden() for label in window.sidebar.findChildren(QLabel)))
            self.assertTrue(all(check.isChecked() for check in window.checks.values()))
        finally:window.close()

    def test_only_raw_sources_and_sidebar_controls_can_shrink(self):
        from PySide6.QtWidgets import QStyleOptionButton,QStyle
        window=AnnotationWindow(DEFAULT)
        try:
            self.assertEqual([window.input_format.itemData(i) for i in range(window.input_format.count())],['lmdb','kitti'])
            self.assertEqual(window.input_format.currentData(),'lmdb')
            window.sidebar.ensurePolished();window.sidebar.layout().activate()
            self.assertLessEqual(window.sidebar.minimumSizeHint().width(),328)
            check=window.checks['instances'];check.resize(250,60);check.fit_text()
            option=QStyleOptionButton();check.initStyleOption(option)
            available=check.style().subElementRect(QStyle.SubElement.SE_CheckBoxContents,option,check).width()
            self.assertTrue(all(check.fontMetrics().horizontalAdvance(line)<=available for line in check.text().splitlines()))
            self.assertIn('复用 SAM mask',check.toolTip())
        finally:window.close()

    def test_view_mode_buttons_keep_one_persistent_selection(self):
        window=AnnotationWindow(DEFAULT)
        try:
            self.assertTrue(window.combined_button.isChecked())
            self.assertEqual(window.track_stage.currentData(),'after')
            with tempfile.TemporaryDirectory() as tmp:
                out=Path(tmp);(out/'raw').mkdir();(out/'raw/index.json').write_text('{}')
                with patch.object(window,'current_context',return_value={'output':tmp}),patch.object(window,'refresh_preview'):
                    for mode in ('semantic','tracks','combined','projection'):
                        button=window.mode_buttons[mode];button.click()
                        self.assertEqual(window.preview_mode,mode)
                        self.assertEqual([key for key,b in window.mode_buttons.items() if b.isChecked()],[mode])
                        button.click();self.assertTrue(button.isChecked())
                    window.show_label_mode('semantic')
                    self.assertTrue(window.mask_button.isChecked())
                with patch.object(window,'current_context',side_effect=ValueError('invalid source')):
                    window.projection_button.click()
                self.assertEqual(window.preview_mode,'semantic')
                self.assertTrue(window.mask_button.isChecked());self.assertFalse(window.projection_button.isChecked())
        finally:window.close()

    def test_result_view_arrow_keys_navigate_frames_without_focusing_the_input(self):
        from PySide6.QtCore import Qt
        from PySide6.QtTest import QTest
        window=AnnotationWindow(DEFAULT)
        try:
            with patch.object(window,'refresh_preview'):
                window.frames.setText('0 2 9');window.result_tabs.setCurrentIndex(1)
                QTest.keyClick(window.result_tabs.tabBar(),Qt.Key.Key_Right)
                self.assertEqual(window.preview_frame.value(),2)
                QTest.keyClick(window.camera_view,Qt.Key.Key_Down)
                self.assertEqual(window.preview_frame.value(),9)
                QTest.keyClick(window.bev_view,Qt.Key.Key_Right)
                self.assertEqual(window.preview_frame.value(),9)
                QTest.keyClick(window.combined_button,Qt.Key.Key_Left)
                self.assertEqual(window.preview_frame.value(),2)
                QTest.keyClick(window.frame_slider,Qt.Key.Key_Up)
                self.assertEqual(window.preview_frame.value(),0)
                window.point_size.setValue(3);QTest.keyClick(window.point_size,Qt.Key.Key_Up)
                self.assertEqual(window.point_size.value(),4);self.assertEqual(window.preview_frame.value(),0)
                window.preview_frame.setValue(2);QTest.keyClick(window.preview_frame,Qt.Key.Key_Up)
                self.assertEqual(window.preview_frame.value(),0)
                window.frames.setText('0-9');window.frames.lineEdit().setCursorPosition(3)
                QTest.keyClick(window.frames.lineEdit(),Qt.Key.Key_Left)
                self.assertEqual(window.frames.lineEdit().cursorPosition(),2);self.assertEqual(window.preview_frame.value(),0)
                window.result_tabs.setCurrentIndex(0)
                QTest.keyClick(window.camera_view,Qt.Key.Key_Right)
                self.assertEqual(window.preview_frame.value(),0)
        finally:window.close()

    def test_advanced_controls_live_in_run_configuration(self):
        window=AnnotationWindow(DEFAULT)
        for control in (window.all_button,window.workers,window.refresh_semantic,window.refresh_classes,window.refresh_instances,window.dry_button):
            self.assertTrue(window.run_config.isAncestorOf(control))
            self.assertFalse(window.sidebar.isAncestorOf(control))
        self.assertTrue(window.sidebar.isAncestorOf(window.run_button))
        window.input_format.setCurrentIndex(window.input_format.findData('lmdb'));window.frames.setText('1-10')
        window.refresh_semantic.setChecked(True);self.assertTrue(window.refresh_classes.isEnabled())
        window.close()

    def test_frame_numbering_starts_at_zero_and_projection_button_is_before_workflow(self):
        window=AnnotationWindow(DEFAULT)
        try:
            self.assertEqual(window.preview_frame.minimum(),0)
            self.assertEqual(window.preview_frame.value(),0)
            self.assertIn('0-9',window.frames.placeholderText())
            self.assertTrue(window.results_panel.isAncestorOf(window.projection_button))
            self.assertFalse(window.source_group.isAncestorOf(window.projection_button))
            self.assertTrue(window.results_panel.isAncestorOf(window.mask_button));self.assertTrue(window.results_panel.isAncestorOf(window.track_button))
            args=annotate_command('/tmp/source','/tmp/job',['semantic'],1,'0-9')
            self.assertEqual(args[args.index('--frames')+1:],list(map(str,range(10))))
        finally:window.close()

    def test_switching_kitti_to_lmdb_resets_preview_to_zero(self):
        window=AnnotationWindow(DEFAULT)
        try:
            window.input_format.setCurrentIndex(window.input_format.findData('kitti'))
            window.preview_frame.setValue(9)
            window.input_format.setCurrentIndex(window.input_format.findData('lmdb'))
            self.assertEqual(window.preview_frame.value(),0)
        finally:window.close()

    def test_preview_frame_is_limited_to_selected_range(self):
        from PySide6.QtCore import Qt
        from PySide6.QtTest import QTest
        from PySide6.QtWidgets import QAbstractSpinBox
        window=AnnotationWindow(DEFAULT)
        try:
            window.frames.setText('0-9')
            spin=window.preview_frame;spin.setValue(9)
            QTest.keyClick(spin,Qt.Key.Key_Down)
            self.assertEqual(spin.value(),9)
            self.assertEqual(spin.maximum(),9)
            self.assertFalse(spin.stepEnabled()&QAbstractSpinBox.StepEnabledFlag.StepDownEnabled)
            spin.setValue(10);self.assertEqual(spin.value(),9)
            window.frames.setText('20-29')
            self.assertEqual((spin.minimum(),spin.maximum()),(20,29))
            self.assertEqual(spin.value(),20)
            window.frames.setText('5')
            self.assertEqual((spin.minimum(),spin.maximum(),spin.value()),(5,5,5))
        finally:window.close()

    def test_preview_frame_bounds_follow_current_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);(out/'raw').mkdir()
            ids=[63,79,824]
            (out/'raw/index.json').write_text(json.dumps({'rows':[{'path':f'{i:06d}.npz','frame_id':i} for i in ids]}))
            window=AnnotationWindow(DEFAULT)
            try:
                window.lmdb_output.setText(str(out));use_cached_work(window,out);window.update_visualization_location()
                self.assertEqual((window.preview_frame.minimum(),window.preview_frame.maximum()),(min(ids),max(ids)))
                window.frames.setText('0-9');self.assertEqual(window.preview_frame.maximum(),9)
                window.frames.clear()
                self.assertEqual((window.preview_frame.minimum(),window.preview_frame.maximum()),(min(ids),max(ids)))
            finally:window.close()

    def test_frame_slider_buttons_and_input_stay_in_sync(self):
        window=AnnotationWindow(DEFAULT)
        try:
            window.frames.setText('0-9')
            self.assertEqual((window.frame_slider.minimum(),window.frame_slider.maximum()),(0,9))
            self.assertEqual(window.frame_count.text(),'帧 / 共 10 帧')
            self.assertIn('原始帧范围：0–9',window.frame_count.toolTip())
            window.frame_slider.setSliderDown(True);window.frame_slider.setSliderPosition(6)
            window.sync_frame_navigation()
            self.assertEqual(window.frame_slider.sliderPosition(),6)
            self.assertEqual(window.preview_frame.value(),0)
            window.frame_slider.setSliderDown(False)
            self.assertEqual(window.preview_frame.value(),6)
            window.frame_slider.setValue(4)
            self.assertEqual(window.preview_frame.value(),4)
            window.next_frame.click();self.assertEqual(window.preview_frame.value(),5)
            window.previous_frame.click();self.assertEqual(window.preview_frame.value(),4)
            window.preview_frame.setValue(8);self.assertEqual(window.frame_slider.value(),8)
            window.preview_frame.setValue(9);self.assertFalse(window.next_frame.isEnabled())
            window.preview_frame.setValue(0);self.assertFalse(window.previous_frame.isEnabled())
            window.frames.setText('5')
            self.assertEqual(window.frame_slider.maximum(),0)
            self.assertFalse(window.previous_frame.isEnabled());self.assertFalse(window.next_frame.isEnabled())
        finally:window.close()

    def test_frame_navigation_skips_missing_frames(self):
        window=AnnotationWindow(DEFAULT)
        try:
            window.frames.setText('63 79 824')
            self.assertEqual(window.frame_slider.maximum(),2)
            window.frame_slider.setValue(1);self.assertEqual(window.preview_frame.value(),79)
            window.next_frame.click();self.assertEqual(window.preview_frame.value(),824)
            window.previous_frame.click();self.assertEqual(window.preview_frame.value(),79)
            window.preview_frame.setValue(80)
            self.assertEqual(window.preview_frame.value(),79)
            self.assertEqual(window.frame_slider.value(),1)
        finally:window.close()

    def test_loading_existing_temporary_directory_limits_frames_without_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);(out/'raw').mkdir()
            (out/'raw/index.json').write_text(json.dumps({'rows':[{'path':f'{i:06d}.npz','original_frame_id':i} for i in range(10)]}))
            from .cli import load_config
            config=load_config(sample_path('testdata/zgc_review/profile.json'))
            config['output']=str(out);config['diagnostic_only']=False
            profile=out/'profile.json';profile.write_text(json.dumps(config))
            window=AnnotationWindow(DEFAULT)
            try:
                window.lmdb_output.setText(str(out));use_cached_work(window,out);window.update_visualization_location()
                self.assertEqual(window.frames.text(),'')
                self.assertEqual(window.preview_frame.maximum(),9)
                window.preview_frame.setValue(9);window.preview_frame.stepDown()
                self.assertEqual(window.preview_frame.value(),9)
                window.frames.setText('0-19')
                with patch.object(window,'current_context',return_value=config):
                    window.update_visualization_location()
                self.assertEqual(window.preview_frame.maximum(),9)
            finally:window.close()

    def test_frame_controls_move_up_to_previous_and_down_to_next(self):
        from PySide6.QtCore import Qt,QPoint,QPointF
        from PySide6.QtGui import QWheelEvent
        from PySide6.QtTest import QTest
        from PySide6.QtWidgets import QAbstractSpinBox
        window=AnnotationWindow(DEFAULT)
        try:
            spin=window.preview_frame;spin.setValue(60)
            QTest.keyClick(spin,Qt.Key.Key_Up);self.assertEqual(spin.value(),59)
            QTest.keyClick(spin,Qt.Key.Key_Down);self.assertEqual(spin.value(),60)
            event=QWheelEvent(QPointF(5,5),QPointF(5,5),QPoint(),QPoint(0,120),Qt.MouseButton.NoButton,Qt.KeyboardModifier.NoModifier,Qt.ScrollPhase.ScrollUpdate,False)
            self.app.sendEvent(spin,event);self.assertEqual(spin.value(),59)
            spin.setValue(0)
            self.assertFalse(spin.stepEnabled()&QAbstractSpinBox.StepEnabledFlag.StepUpEnabled)
            self.assertTrue(spin.stepEnabled()&QAbstractSpinBox.StepEnabledFlag.StepDownEnabled)
            QTest.keyClick(spin,Qt.Key.Key_Down);self.assertEqual(spin.value(),1)
            spin.setValue(spin.maximum());self.assertTrue(spin.stepEnabled()&QAbstractSpinBox.StepEnabledFlag.StepUpEnabled)
            self.assertFalse(spin.stepEnabled()&QAbstractSpinBox.StepEnabledFlag.StepDownEnabled)
            window.point_size.setValue(3);window.point_size.stepUp();self.assertEqual(window.point_size.value(),4)
        finally:window.close()

    def test_frame_selection_is_on_left_without_persistent_help_text(self):
        window=AnnotationWindow(DEFAULT)
        try:
            self.assertTrue(window.sidebar.isAncestorOf(window.frames))
            self.assertFalse(window.run_config.isAncestorOf(window.frames))
            self.assertFalse(hasattr(window,'frames_help'))
            self.assertTrue(window.frames.isEditable())
            window.input_format.setCurrentIndex(window.input_format.findData('lmdb'))
            args=annotate_command('/tmp/source','/tmp/job',['semantic'],1,'1-10')
            self.assertEqual(args[args.index('--frames')+1:],list(map(str,range(1,11))))
        finally:window.close()

    def test_subset_uses_separate_work_and_explicit_public_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            window=AnnotationWindow(DEFAULT)
            try:
                window.lmdb_root.setText(str(sample_path('testdata/lmdb_sample')))
                window.frames.setText('63 79');window.select_stages(['semantic','render','catalog'])
                context=window.current_context()
                self.assertIn('/.lidar_annotator_work/',context['output'])
                self.assertTrue(window.lmdb_root.isEnabled())
                with patch.object(window.process,'start') as start:
                    window.start(True)
                args=start.call_args.args[1]
                self.assertIn('annotate',args)
                self.assertEqual(args[args.index('--output')+1],context['dataset_output']);self.assertEqual(args[args.index('--work-dir')+1],context['output'])
                self.assertIn('标准数据目录',window.log.toPlainText())
                self.assertEqual(args[args.index('--frames')+1:args.index('--dry-run')],['63','79'])
            finally:window.close()

    def test_projection_button_loads_raw_projection_without_models(self):
        from .test_lmdb_input import fixture
        from .lmdb_input import read_lmdb
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=fixture(root);out=root/'job';read_lmdb(source,out/'raw')
            window=AnnotationWindow(DEFAULT)
            try:
                window.input_format.setCurrentIndex(window.input_format.findData('lmdb'));window.lmdb_output.setText(str(out))
                window.projection_button.click()
                self.wait_preview(window)
                self.assertEqual(window.camera_view.data['frame'],0)
                self.assertTrue(window.results_panel.isAncestorOf(window.projection_button))
                self.assertFalse((out/'projection/000000.png').exists())
                self.assertEqual(window.result_tabs.currentIndex(),1)
            finally:window.close()

    def test_projection_button_can_read_one_frame_before_processing(self):
        from .test_lmdb_input import fixture
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=fixture(root);out=root/'job'
            window=AnnotationWindow(DEFAULT)
            try:
                window.input_format.setCurrentIndex(window.input_format.findData('lmdb'));window.lmdb_root.setText(str(source));window.lmdb_output.setText(str(out))
                window.projection_button.click()
                self.wait_preview(window)
                self.assertEqual(window.camera_view.data['frame'],0)
                self.assertTrue(window.results_panel.isAncestorOf(window.projection_button))
                self.assertFalse((out/'projection/000000.png').exists())
                self.assertFalse((out/'run_report.json').exists())
            finally:window.close()

    def wait_preview(self,window):
        deadline=time.monotonic()+10
        while time.monotonic()<deadline:
            self.app.processEvents()
            if window.frame_reader is None and window.camera_view.data is not None:return
            time.sleep(.01)
        self.fail(window.preview_status.text())

    def test_current_frame_reloads_latest_mask_without_saved_preview(self):
        from .test_lmdb_input import fixture
        from .lmdb_input import read_lmdb
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=fixture(root);out=root/'job';read_lmdb(source,out/'raw')
            stale=out/'review/visualization/000000.png';stale.parent.mkdir(parents=True);stale.write_text('not a usable preview')
            window=AnnotationWindow(DEFAULT)
            try:
                window.input_format.setCurrentIndex(window.input_format.findData('lmdb'));window.lmdb_output.setText(str(out));use_cached_work(window,out);window.result_tabs.setCurrentIndex(1);window.load_image();self.wait_preview(window)
                self.assertFalse(window.camera_view.data['semantic'])
                before=window.camera_view.data['image'].pixelColor(0,0)
                mask=out/'data/labels/zgc/image_masks/000000.png';mask.parent.mkdir(parents=True);Image.new('L',(64,48),3).save(mask)
                window.preview_timer.timeout.emit();self.wait_preview(window)
                self.assertTrue(window.camera_view.data['semantic'])
                self.assertNotEqual(before,window.camera_view.data['image'].pixelColor(0,0))
                self.assertLessEqual(len(window.camera_view.groups),32)
                self.assertFalse(hasattr(window,'read_lmdb'))
            finally:window.close()

    def test_live_reader_supports_cached_original_frame_ids(self):
        from .cli import load_config
        from .live_preview import load_frame
        data=load_frame(load_config(sample_path('testdata/zgc_review/profile.json')),79)
        self.assertEqual(data['frame'],79)
        self.assertGreater(len(data['camera_points']),0)
        self.assertEqual(len(data['centres']),2)
        self.assertGreater(len(data['boxes']),0)

    def test_target_annotations_draw_only_numeric_ids(self):
        from .cli import load_config
        from .live_preview import load_frame,PointCanvas
        data=load_frame(load_config(sample_path('testdata/zgc_review/profile.json')),79)
        data['records']=[{'prompt':'person','display_id':0,'bbox_xyxy':[10,10,20,30]},
                         {'prompt':'car','display_id':17,'bbox_xyxy':[30,10,50,30]},
                         {'prompt':'person','display_id':None,'bbox_xyxy':[60,10,70,30]}]
        for bev in (False,True):
            canvas=PointCanvas(bev=bev);canvas.data=data
            try:
                with patch('tools.lidar_annotator.live_preview.QPainter') as painter:
                    canvas.paintEvent(None)
                    texts=[call.args[-1] for call in painter.return_value.drawText.call_args_list]
                annotations=[text for text in texts if text.isdigit()]
                expected=([str(int(tid)) for tid in data['centre_ids']]+[str(tid) for tid in data['box_ids']]) if bev else ['0','17']
                self.assertEqual(annotations,expected)
            finally:canvas.close()

    def test_semantic_legend_uses_the_rendered_recipe_and_hides_in_projection(self):
        from .cli import load_config
        from .live_preview import load_frame
        config=load_config(sample_path('testdata/zgc_review/profile.json'))
        data=load_frame(config,79,mode='semantic')
        self.assertTrue(data['semantic_classes'])
        window=AnnotationWindow(DEFAULT)
        try:
            window.update_semantic_legend(data['semantic_classes'])
            self.assertFalse(window.semantic_legend.isHidden())
            for item in data['semantic_classes']:
                self.assertIn('#{:02X}{:02X}{:02X}'.format(*item['rgb']),window.semantic_legend.text())
            self.assertIn('行人',window.semantic_legend.text())
            self.assertIn('立柱',window.semantic_legend.text())
            window.update_semantic_legend([{'name':'custom <class>','rgb':[12,34,56]}])
            self.assertIn('#0C2238',window.semantic_legend.text())
            self.assertIn('custom&nbsp;&lt;class&gt;',window.semantic_legend.text())
            projection=load_frame(config,79,labels=False,mode='projection')
            window.update_semantic_legend(projection['semantic_classes'])
            self.assertTrue(window.semantic_legend.isHidden())
        finally:window.close()

    def test_view_range_filters_both_views_before_display_and_is_editable(self):
        import numpy as np
        from .test_lmdb_input import fixture
        from .lmdb_input import read_lmdb
        from .live_preview import load_frame
        from .display_range import point_mask
        points=np.array([[0,0,0],[20,-20,-3],[-20,20,6],[20.01,0,0],[0,0,-3.01],[0,0,6.01],[0,0,float('nan')]],float)
        original=points.copy()
        np.testing.assert_array_equal(point_mask(points),[True,True,True,False,False,False,False])
        np.testing.assert_array_equal(points,original)
        with self.assertRaises(ValueError):point_mask(points,(20,6,-3))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=fixture(root);out=root/'job';read_lmdb(source,out/'raw')
            path=out/'raw/frames/000000.npz'
            with np.load(path) as data:arrays={k:data[k] for k in data.files}
            arrays.update(xyz=points,pixels_uv=np.tile([10,10],(len(points),1)),camera_depth=np.ones(len(points))*5,in_image=np.ones(len(points),bool))
            np.savez(path,**arrays)
            payload=load_frame(dict(output=str(out)),0)
            self.assertEqual(len(payload['camera_points']),3);self.assertEqual(len(payload['bev_points']),3)
            self.assertGreater(payload['palette'][0,0],payload['palette'][0,2])
            self.assertGreater(payload['palette'][-1,2],payload['palette'][-1,0])
            window=AnnotationWindow(DEFAULT)
            try:
                window.input_format.setCurrentIndex(window.input_format.findData('lmdb'));window.lmdb_output.setText(str(out));use_cached_work(window,out);self.assertEqual(window.preview_limits(),(20,-3,6))
                window.result_tabs.setCurrentIndex(1);window.load_image();self.wait_preview(window)
                window.xy_extent.setValue(10);self.wait_preview(window)
                self.assertEqual(window.camera_view.data['limits'],(10,-3,6))
                self.assertEqual(len(window.camera_view.data['camera_points']),1)
                self.assertEqual(len(window.bev_view.data['bev_points']),1)
                self.assertEqual(window.bev_view.transform_values()[1:],(20,20))
                with np.load(path) as data:np.testing.assert_array_equal(data['xyz'],points)
            finally:window.close()

    def test_right_view_modes_show_current_mask_and_saved_track_ids(self):
        from .test_lmdb_input import fixture
        from .lmdb_input import read_lmdb
        from PIL import Image
        import json
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=fixture(root);out=root/'job';read_lmdb(source,out/'raw')
            mask=out/'data/labels/zgc/image_masks/000000.png';mask.parent.mkdir(parents=True);Image.new('L',(64,48),3).save(mask)
            record=out/'data/visual_instances/000000.json';record.parent.mkdir(parents=True);record.write_text(json.dumps(dict(records=[dict(prompt='person',track_id=42,bbox_xyxy=[2,3,20,30])])))
            window=AnnotationWindow(DEFAULT)
            try:
                window.input_format.setCurrentIndex(window.input_format.findData('lmdb'));window.lmdb_output.setText(str(out));use_cached_work(window,out)
                window.mask_button.click();self.wait_preview(window)
                self.assertEqual(window.camera_view.data['mode'],'semantic');self.assertTrue(window.camera_view.data['semantic'])
                self.assertEqual(window.camera_view.data['image'].pixelColor(0,0).getRgb()[:3],(65,135,240))
                self.assertEqual(len(window.camera_view.data['camera_points']),0)
                window.track_button.click();self.wait_preview(window)
                self.assertEqual(window.camera_view.data['mode'],'tracks');self.assertFalse(window.camera_view.data['semantic'])
                self.assertEqual(window.camera_view.data['records'][0]['display_id'],42)
                window.projection_button.click();self.wait_preview(window)
                self.assertEqual(window.camera_view.data['mode'],'projection');self.assertFalse(window.camera_view.data['records'])
                self.assertIn('原始深度投影',window.preview_status.text())
                self.assertNotIn('尚未生成',window.preview_status.text())
            finally:window.close()

    def test_printed_directory_is_clickable_without_duplicate_lines(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)/'图像 & previews';folder.mkdir();window=AnnotationWindow(DEFAULT)
            text='可视化目录：'+str(folder)+'\n'
            window.log.append_stream(text[:12]);window.log.append_stream(text[12:]);window.log.append_directory(folder)
            self.assertEqual(window.log.toPlainText().count('可视化目录：'),1);self.assertIn('href=',window.log.toHtml())
            window.log.append_stream('RuntimeError: <real failure>\n')
            self.assertIn('RuntimeError: <real failure>',window.log.toPlainText())
            with patch('tools.lidar_annotator.qt_app.QDesktopServices.openUrl') as opened:
                window.log.anchorClicked.emit(QUrl.fromLocalFile(str(folder)));self.assertEqual(opened.call_args.args[0].toLocalFile(),str(folder))
                window.log.anchorClicked.emit(QUrl.fromLocalFile(str(folder/'not-created')));self.assertEqual(opened.call_count,1)
            self.assertIn('尚未生成',window.state.text());window.close()

    def test_current_stage_visualization_path_and_open_button(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);window=AnnotationWindow(DEFAULT);window.input_format.setCurrentIndex(window.input_format.findData('lmdb'));window.lmdb_output.setText(str(out));use_cached_work(window,out);window.update_visualization_location('semantic')
            self.assertTrue(window.results_panel.isAncestorOf(window.visualization_path));self.assertFalse(window.sidebar.isAncestorOf(window.visualization_path))
            self.assertIs(window.results_panel.layout().itemAt(0).widget(),window.visualization_panel)
            self.assertEqual(window.open_stage_button.text(),'打开目录')
            expected=out/'data/labels/zgc/overlay'
            self.assertEqual(window.visualization_path.text(),str(expected));self.assertFalse(window.open_stage_button.isEnabled());self.assertIn('尚未生成',window.visualization_note.text())
            expected.mkdir(parents=True);(expected/'000000.jpg').touch();window.update_visualization_location('semantic')
            self.assertTrue(window.open_stage_button.isEnabled());window.copy_stage_button.click();self.assertEqual(self.app.clipboard().text(),str(expected))
            with patch('tools.lidar_annotator.qt_app.QDesktopServices.openUrl') as opened:
                window.open_stage_button.click();self.assertEqual(opened.call_args.args[0].toLocalFile(),str(expected))
            window.update_visualization_location('instances');self.assertEqual(window.visualization_path.text(),'');self.assertFalse(window.open_stage_button.isEnabled());self.assertIn('联合图将在',window.visualization_note.text());window.close()

    def test_lmdb_recipe_refresh_controls_and_debug_range(self):
        window=AnnotationWindow(DEFAULT);window.input_format.setCurrentIndex(window.input_format.findData('lmdb'))
        self.assertTrue(window.profile.isEnabled());self.assertTrue(window.recipe_button.isEnabled());self.assertTrue(window.refresh_semantic.isEnabled())
        window.refresh_semantic.setChecked(True);self.assertTrue(window.refresh_classes.isEnabled())
        self.assertTrue(window.results_panel.isAncestorOf(window.progress));self.assertTrue(window.results_panel.isAncestorOf(window.frame_progress))
        args=annotate_command('/tmp/source','/tmp/job',['semantic'],1,'60-63 79',recipe_path=window.profile.text(),refresh_semantic=True,semantic_classes=[3])
        self.assertEqual(args[args.index('--frames')+1:args.index('--refresh-semantic')],['60','61','62','63','79'])
        self.assertIn('--recipe',args);self.assertEqual(args[-2:],['--semantic-classes','3']);window.close()

    def test_temporal_pipeline_is_one_checkbox_for_all_substeps(self):
        from .qt_app import TEMPORAL_FLOW
        window=AnnotationWindow(DEFAULT);window.input_format.setCurrentIndex(window.input_format.findData('lmdb'));window.select_stages([])
        window.temporal_flow.setChecked(True)
        self.assertTrue(all(window.checks[s].isChecked() and window.checks[s].isHidden() for s in TEMPORAL_FLOW))
        window.temporal_flow.setChecked(False);self.assertFalse(any(window.checks[s].isChecked() for s in TEMPORAL_FLOW));window.close()

    def test_labels_and_visualizations_are_one_output_option(self):
        from .qt_app import EXPORT_FLOW
        window=AnnotationWindow(DEFAULT);window.input_format.setCurrentIndex(window.input_format.findData('lmdb'))
        self.assertEqual(window.output_group.title(),'输出结果');self.assertEqual(window.export_flow.text(),'生成标签与可视化')
        self.assertTrue(all(window.checks[s].isHidden() for s in EXPORT_FLOW))
        window.export_flow.setChecked(False);self.assertFalse(any(window.checks[s].isChecked() for s in EXPORT_FLOW))
        window.export_flow.setChecked(True);self.assertTrue(all(window.checks[s].isChecked() for s in EXPORT_FLOW));window.close()

    def test_workers_only_enable_for_visualization(self):
        window=AnnotationWindow(DEFAULT);window.input_format.setCurrentIndex(window.input_format.findData('lmdb'))
        window.select_stages(['semantic']);self.assertFalse(window.workers.isEnabled())
        window.checks['render'].setChecked(True);self.assertTrue(window.workers.isEnabled())
        window.checks['render'].setChecked(False);self.assertFalse(window.workers.isEnabled());window.close()

    def test_partial_frame_commands_use_shared_cli(self):
        command=command_for(DEFAULT,['render'],2,'79,824')
        self.assertEqual(command[-3:],['--frames','79','824'])
        with self.assertRaises(ValueError):command_for(DEFAULT,['optimize'],2,'79')

    def test_data_configuration_is_in_left_sidebar(self):
        window=AnnotationWindow(DEFAULT)
        self.assertTrue(window.sidebar.isAncestorOf(window.profile));self.assertFalse(window.sidebar.isAncestorOf(window.summary));self.assertTrue(window.help_page.isAncestorOf(window.summary));window.close()

    def test_start_button_writes_portable_data_at_the_entered_directory(self):
        from .test_lmdb_input import fixture
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=fixture(root);public=root/'export';public.mkdir()
            (public/'job.json').write_text(json.dumps(dict(schema='lidar_fresh_job_v1',source=str(source),frame_ids=[0])))
            (public/'raw').mkdir();(public/'raw/old.txt').write_text('preserve old task')
            window=AnnotationWindow(DEFAULT)
            try:
                window.lmdb_root.setText(str(source));window.lmdb_output.setText(str(public));window.select_stages(['render','catalog'])
                work=Path(window.base_context()['output']);self.assertNotEqual(work,public)
                window.run_button.click();deadline=time.monotonic()+15
                while time.monotonic()<deadline:
                    self.app.processEvents()
                    if window.process.state()==QProcess.ProcessState.NotRunning and window.run_button.isEnabled():break
                    time.sleep(.01)
                self.assertIn('处理完成',window.state.text(),window.log.toPlainText())
                for name in ('image_2','velodyne','calib.txt','times.txt','poses.txt','metadata.npz'):self.assertTrue((public/name).exists(),name)
                for name in ('raw','review','job.json'):self.assertFalse((public/name).exists(),name)
                self.assertTrue((work/'raw/index.json').exists());self.assertIn('标准数据目录',window.log.toPlainText())
                with patch('tools.lidar_annotator.qt_app.QDesktopServices.openUrl') as opened:
                    window.open_result();self.assertEqual(opened.call_args.args[0].toLocalFile(),str(public))
                self.assertEqual([p.read_text() for p in work.parent.glob('*_legacy_*/raw/old.txt')],['preserve old task'])
            finally:window.close()

    def test_lmdb_open_output_never_exposes_internal_work_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            job=Path(tmp);(job/'raw').mkdir();(job/'raw/status.json').write_text(json.dumps(dict(status='running',completed=250,total=971)))
            window=AnnotationWindow(DEFAULT)
            try:
                with patch.object(window,'current_context',return_value={'output':str(job),'input_format':'lmdb'}),patch('tools.lidar_annotator.qt_app.QDesktopServices.openUrl') as opened:
                    window.open_result();opened.assert_not_called();self.assertIn('250/971',window.state.text())
                    (job/'raw/status.json').write_text(json.dumps(dict(status='running',phase='kitti',completed=25,total=971)))
                    window.open_result();opened.assert_not_called();self.assertIn('转换 KITTI：25/971',window.state.text())
                    (job/'raw/status.json').write_text(json.dumps(dict(status='complete',kitti_export_status='failed',kitti_export_error='disk full')))
                    window.open_result();opened.assert_not_called();self.assertIn('导出失败：disk full',window.state.text())
                    portable=job/'kitti';portable.mkdir();(portable/'metadata.npz').touch()
                    window.open_result();self.assertEqual(opened.call_args.args[0].toLocalFile(),str(portable))
            finally:window.close()

    def test_open_output_prefers_exported_dataset_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'job';out.mkdir();labels=Path(tmp)/'sequence/labels';labels.mkdir(parents=True)
            (out/'label_export.json').write_text(json.dumps({'output':str(labels),'status':'complete'}))
            window=AnnotationWindow(DEFAULT)
            try:
                with patch.object(window,'current_context',return_value={'output':str(out)}),patch('tools.lidar_annotator.qt_app.QDesktopServices.openUrl') as opened:
                    window.open_result();self.assertEqual(opened.call_args.args[0].toLocalFile(),str(labels))
            finally:window.close()

    def test_application_icon_is_loaded_from_svg(self):
        from PySide6.QtGui import QIcon
        from xml.etree import ElementTree
        from tools.lidar_annotator.qt_app import AnnotationWindow
        ElementTree.parse(ICON_PATH)
        icon=QIcon(str(ICON_PATH))
        self.assertFalse(icon.isNull())
        window=AnnotationWindow(DEFAULT);self.assertFalse(window.windowIcon().isNull());window.close()

    def test_desktop_installer_quotes_paths_with_spaces(self):
        from scripts.install_desktop import install
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'Project With Spaces'
            applications=Path(tmp)/'User Apps'
            target=install(applications,root)
            content=target.read_text()
            self.assertIn(f'Exec="{root.resolve()}/run.sh" gui',content)
            self.assertIn(f'Icon={root.resolve()}/assets/icons/lidar-annotator.svg'.replace(' ','\\s'),content)
            self.assertIn('Terminal=false',content)

    def test_preview_has_independent_views_and_point_size(self):
        window=AnnotationWindow(DEFAULT)
        try:
            self.assertIsNot(window.camera_view,window.bev_view)
            window.point_size.setValue(5)
            self.assertEqual(window.camera_view.point_size,5);self.assertEqual(window.bev_view.point_size,5)
            window.camera_view.zoom=3
            self.assertEqual(window.bev_view.zoom,1)
        finally:window.close()

    def test_lmdb_dropdown_enables_full_workflow(self):
        window=AnnotationWindow(DEFAULT);window.input_format.setCurrentIndex(window.input_format.findData('lmdb'))
        self.assertFalse(hasattr(window,'read_lmdb'));self.assertTrue(window.lmdb_root.isEnabled());self.assertTrue(all(check.isEnabled() and check.isChecked() for check in window.checks.values()));self.assertTrue(window.workers.isEnabled());window.close()
        args=annotate_command('/tmp/recording','outputs/lmdb',['semantic','render'],3,'63 79')
        self.assertIn('annotate',args);self.assertEqual(args[args.index('--stages')+1:args.index('--workers')],['semantic','render']);self.assertEqual(args[args.index('--frames')+1:],['63','79'])

    def test_lmdb_selection_dispatches_annotate_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            window=AnnotationWindow(DEFAULT);window.input_format.setCurrentIndex(window.input_format.findData('lmdb'));window.lmdb_root.setText(str(sample_path('testdata/lmdb_sample')));window.lmdb_output.setText(str(Path(tmp)/'export'));window.frames.setText('63');window.start(True)
            deadline=time.monotonic()+10
            while time.monotonic()<deadline:
                self.app.processEvents()
                if window.process.state()==QProcess.ProcessState.NotRunning and window.run_button.isEnabled():break
                time.sleep(.01)
            self.assertIn('annotate',window.log.toPlainText());self.assertIn('--stages',window.log.toPlainText());self.assertIn('--workers',window.log.toPlainText());self.assertFalse((Path(tmp)/'export').exists());self.assertTrue(window.run_button.isEnabled());window.close()

    def test_frame_progress_reads_structured_status_without_log_spam(self):
        import json
        from tools.autolabel.task_progress import report_progress
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);(out/'logs').mkdir()
            (out/'run_report.json').write_text(json.dumps(dict(status='running',stages=[dict(stage='semantic',status='running')])))
            (out/'logs/semantic.log').write_text('warning retained\nfresh_semantics 3/971 frame_id=2\n')
            report_progress('semantic',3,971,frame=2,job=out)
            window=AnnotationWindow(DEFAULT);window.active_config=dict(output=str(out));window.selected_stages=['semantic'];window.read_stage_log()
            self.assertEqual(window.frame_progress.maximum(),971);self.assertEqual(window.frame_progress.value(),3)
            self.assertIn('帧 2',window.state.text());self.assertIn('warning retained',window.log.toPlainText());self.assertNotIn('fresh_semantics',window.log.toPlainText())
            report_progress('instances',8,10,job=out);window.read_stage_log();self.assertEqual(window.frame_progress.maximum(),0);window.close()

    def test_startup_uses_raw_source_instead_of_bundled_cached_profile(self):
        profile=sample_path('testdata/zgc_review/profile.json')
        with tempfile.TemporaryDirectory() as temp,patch.dict(os.environ,{'XDG_CONFIG_HOME':temp}):
            window=AnnotationWindow(profile)
            try:
                self.assertEqual(window.input_format.currentData(),'lmdb')
                self.assertTrue(window.profile.isHidden());self.assertTrue(window.profile_browse.isHidden())
                self.assertEqual(window.preview_frame.value(),0)
                self.assertTrue(all(check.isChecked() for check in window.checks.values()))
            finally:window.close()

    def test_relative_source_uses_project_root(self):
        with patch('os.getcwd',return_value='/tmp'):
            window=AnnotationWindow(DEFAULT)
            try:
                window.lmdb_root.setText('../recording')
                command=annotate_command(window.lmdb_root.text(),'outputs/test',['render'],2)
                self.assertEqual(Path(command[command.index('--source')+1]).resolve(),(REPO/'../recording').resolve())
            finally:window.close()

    def test_dry_run_restores_controls_and_emits_cli_plan(self):
        window=AnnotationWindow(DEFAULT);window.start(True)
        deadline=time.monotonic()+10
        while time.monotonic()<deadline:
            self.app.processEvents()
            if window.process.state()==QProcess.ProcessState.NotRunning and window.run_button.isEnabled():break
            time.sleep(.01)
        self.assertTrue(window.run_button.isEnabled());self.assertFalse(window.stop_button.isEnabled())
        self.assertIn('dry_run',window.log.toPlainText());self.assertIn('处理完成',window.state.text());window.close()


if __name__=='__main__':unittest.main()
