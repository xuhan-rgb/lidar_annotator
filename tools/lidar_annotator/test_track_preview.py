"""Synthetic geometry and offscreen controls for instance point-cloud inspection."""
import hashlib,json,os,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import numpy as np
from PIL import Image
from PySide6.QtCore import QPoint,QPointF,Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QApplication
from .live_preview import load_frame,PointCanvas,fingerprint
from .track_preview import track_view
from .qt_app import AnnotationWindow,FrameSlider
from .cli import DEFAULT


def fixture(root):
    raw=root/'raw';raw.mkdir();(raw/'frames').mkdir()
    Image.new('RGB',(8,8),(100,100,100)).save(raw/'image.png')
    (raw/'index.json').write_text(json.dumps({'rows':[{'path':'frames/000000.npz','original_frame_id':63,'image_path':str(raw/'image.png')}]}))
    xyz=np.array([[0,0,5],[.1,0,5],[1,0,5],[.5,0,5],[-1,0,5],[0,0,-1]],np.float32)
    intrinsic=np.array([[10,0,4],[0,10,4],[0,0,1]],float)
    uv=xyz@intrinsic.T;uv=uv[:,:2]/uv[:,2:3]
    np.savez(raw/'frames/000000.npz',xyz=xyz,pixels_uv=uv,visible=np.array([1,1,1,1,1,0],bool),camera_depth=xyz[:,2],cam_intrinsic=intrinsic,cam_extrinsic=np.eye(4),point_raw_labels=np.array([1,1,3,2,0,255],np.uint8))
    visual=root/'data/visual_instances';visual.mkdir(parents=True)
    masks=np.zeros((2,8,8),bool);masks[0,4,4:6]=True;masks[1,4,5:7]=True
    np.savez(visual/'000000.npz',mask_shape=[8,8],masks_packed=np.packbits(masks.reshape(2,-1),axis=1,bitorder='little'))
    (visual/'000000.json').write_text(json.dumps({'records':[{'prompt':'person','track_id':101,'bbox_xyxy':[4,4,6,5]},{'prompt':'person','track_id':202,'bbox_xyxy':[5,4,7,5]}]}))
    person=root/'data/person/metadata';person.mkdir(parents=True)
    (person/'000000.json').write_text(json.dumps({'instances':[
        dict(sam_source_instance_id=tid,status='accepted_xy_from_all_exclusive_mask_points',
             person_class_confirmed=True,yolo_match={'score':.9,'bbox_xyxy':[3,3,7,6]}) for tid in (101,202)]}))
    video=root/'temporal/map_review/sam3_full_video';video.mkdir(parents=True)
    (video/'native_to_display_ids.json').write_text(json.dumps({'101':7,'202':9}))
    centers=video/'interior_half_review/consistent_tracks';centers.mkdir(parents=True)
    np.savez(centers/'centers_only.npz',frame_ids=[0,0],sam_video_ids=[9,7],position_available=[False,True],original_point_lidar=[[1,0,5],[.05,0,5]],corrected_point_lidar=[[np.nan]*3,[.05,.05,5.1]])
    semantic=root/'data/labels/zgc/image_masks';semantic.mkdir(parents=True)
    Image.new('L',(8,8),1).save(semantic/'000000.png')
    return {'output':str(root),'review_output':str(root/'review')}


class TrackPreviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])

    def test_combined_hides_unconfirmed_candidates_but_tracks_keeps_them(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);context=fixture(root)
            path=root/'data/person/metadata/000000.json'
            meta=json.loads(path.read_text());meta['instances'][1]['yolo_match']=None
            meta['instances'][1]['person_class_confirmed']=False
            path.write_text(json.dumps(meta))
            data=load_frame(context,63,mode='combined')
            self.assertEqual([r['display_id'] for r in data['records']],[7])
            self.assertEqual(data['records'][0]['bbox_xyxy'],[3,3,7,6])
            self.assertEqual(data['records'][0]['bbox_source'],'yolo26x')
            self.assertEqual([t['id'] for t in data['targets']],[7])
            self.assertEqual([r['display_id'] for r in load_frame(context,63,mode='tracks')['records']],[7,9])
            path.unlink()
            data=load_frame(context,63,mode='combined')
            self.assertEqual(data['records'],[]);self.assertEqual(data['targets'],[])
            self.assertEqual(len(data['centres']),0)

    def test_combined_vehicle_uses_yolo_box_and_saved_current_frame_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);context=fixture(root);visual=root/'data/visual_instances'
            path=visual/'000000.json';meta=json.loads(path.read_text())
            meta['records'].append(dict(prompt='car',track_id=303,bbox_xyxy=[1,1,3,3]))
            path.write_text(json.dumps(meta))
            with np.load(visual/'000000.npz') as z:
                packed=np.r_[z['masks_packed'],np.zeros((1,8),np.uint8)]
            np.savez(visual/'000000.npz',mask_shape=[8,8],masks_packed=packed)
            vehicle=root/'data/vehicle/refined';vehicle.mkdir(parents=True)
            (vehicle/'000000.json').write_text(json.dumps({'all_detections':[
                dict(prediction_index=4,status='review_label',sam_bbox_xyxy=[1,1,3,3],
                     class_verification={'yolo_score':.8,'yolo_bbox_xyxy':[.5,.5,3.5,3.5]}),
                dict(prediction_index=5,status='ignore_unconfirmed',class_verification=None)]}))
            np.savez(vehicle/'000000.npz',seed_ids=[4,5],box_3d=[[0,0,1,4,2,2,0],[10,10,1,4,2,2,0]])
            data=load_frame(context,63,mode='combined')
            self.assertEqual(data['records'][-1]['bbox_xyxy'],[.5,.5,3.5,3.5])
            self.assertEqual(data['records'][-1]['bbox_source'],'yolo26x')
            self.assertEqual(data['box_ids'],[4])
            np.testing.assert_allclose(data['boxes'][0],[[-2,-1],[2,-1],[2,1],[-2,1]])
            (vehicle/'000000.json').unlink()
            self.assertEqual(load_frame(context,63,mode='combined')['boxes'],[])

    def test_mask_exclusivity_and_saved_centres_match_ids_and_local_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            context=fixture(Path(tmp));data=load_frame(context,63,mode='tracks')
            self.assertEqual([t['id'] for t in data['targets']],[7,9])
            a,b=data['targets'];np.testing.assert_array_equal(a['indices'],[0,1]);np.testing.assert_array_equal(b['indices'],[2])
            np.testing.assert_allclose(a['original'],[.05,0,5]);np.testing.assert_allclose(a['corrected'],[.05,.05,5.1])
            self.assertIsNone(b['corrected'])
            with np.load(Path(tmp)/'raw/frames/000000.npz') as z:original=z['xyz'].copy()
            before=track_view(data,7,'before');after=track_view(data,7,'after');compare=track_view(data,7,'compare')
            np.testing.assert_allclose(before['track_layers'][0]['bev'],original[:2,:2])
            np.testing.assert_allclose(after['track_layers'][0]['bev'],original[:2,:2])
            np.testing.assert_allclose(after['track_layers'][0]['camera'],[[4,4],[4.2,4]])
            self.assertEqual(len(after['bev_points']),len(data['bev_points'])-2)
            self.assertEqual(len(compare['track_layers']),2);self.assertEqual(len(compare['track_arrows']),1)
            self.assertEqual(compare['track_layers'][0]['color'],(155,155,155))
            np.testing.assert_allclose(before['focus_bev'],after['focus_bev'])
            np.testing.assert_array_equal(a['points'],original[:2])

    def test_missing_optimization_and_empty_id_are_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            data=load_frame(fixture(Path(tmp)),63,mode='tracks')
            result=track_view(data,9,'after');self.assertEqual(result['track_layers'],[])
            self.assertIn('无可用优化结果',result['status'])
            self.assertEqual(len(track_view(data,9,'before')['track_layers']),1)
            result=track_view(data,99,'compare');self.assertIn('当前帧无此 ID',result['status']);self.assertIsNone(result['focus_bev'])

    def test_tracking_preserves_original_image_colors_outside_instance_overlay(self):
        with tempfile.TemporaryDirectory() as tmp:
            data=load_frame(fixture(Path(tmp)),63,mode='tracks')
            for stage in ('before','after','compare'):
                result=track_view(data,7,stage)
                self.assertEqual(result['image'].pixelColor(0,0).getRgb()[:3],tuple(data['track_rgb'][0,0]))
                self.assertEqual(result['image'].pixelColor(6,4).getRgb()[:3],tuple(data['track_rgb'][4,6]))
                expected=(data['track_rgb'][4,4]*.65+np.array([0,255,255])*.35).astype(np.uint8)
                self.assertEqual(result['image'].pixelColor(4,4).getRgb()[:3],tuple(expected))

    def test_optimized_position_filters_outliers_without_moving_measurements(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);context=fixture(root);path=root/'raw/frames/000000.npz'
            with np.load(path) as z:arrays={k:z[k] for k in z.files}
            arrays['xyz']=np.r_[arrays['xyz'],[[0,0,5.9],[.49,0,5],[-.03,0,5],[.02,.01,5],[0,-.02,5],[2,0,5]]].astype(np.float32)
            arrays['pixels_uv']=np.r_[arrays['pixels_uv'],[[4,4],[4.98,4],[4,4],[4,4],[4,4],[4,4]]]
            arrays['camera_depth']=np.r_[arrays['camera_depth'],5.9,5,5,5,5,5]
            arrays['visible']=np.r_[arrays['visible'],[True]*6]
            arrays['point_raw_labels']=np.r_[arrays['point_raw_labels'],[1]*6].astype(np.uint8)
            np.savez(path,**arrays)
            centers=root/'temporal/map_review/sam3_full_video/interior_half_review/consistent_tracks/centers_only.npz'
            with np.load(centers) as z:values={k:z[k] for k in z.files}
            values['corrected_point_lidar'][1]=[-.1,.05,5.1];np.savez(centers,**values)
            data=load_frame(context,63,mode='tracks');after=track_view(data,7,'after')
            # A coherent body is retained despite height changes or a slightly offset centre.
            np.testing.assert_allclose(after['track_layers'][0]['bev'],[[0,0],[.1,0],[0,0],[.49,0],[-.03,0],[.02,.01],[0,-.02]])
            joint=load_frame(context,63,mode='combined')
            np.testing.assert_array_equal(joint['bev_points'],arrays['xyz'][:,:2])
            np.testing.assert_array_equal(joint['bev_colors'][0],[0,255,255])
            np.testing.assert_array_equal(joint['bev_colors'][7],[0,255,255])
            np.testing.assert_array_equal(joint['bev_colors'][11],[100,100,100])
            self.assertEqual(len(data['targets'][0]['rejected_indices']),1)
            # A conflicting trajectory centre cannot erase the current body points.
            values['corrected_point_lidar'][1]=[3,3,5.1];np.savez(centers,**values)
            joint=load_frame(context,63,mode='combined')
            np.testing.assert_array_equal(joint['bev_colors'][[0,1,6,7]],[[0,255,255]]*4)
            self.assertTrue(joint['targets'][0]['semantic_selection']['review_required'])
            np.testing.assert_array_equal(joint['bev_points'],arrays['xyz'][:,:2])

    def test_combined_keeps_measurements_and_semantic_colors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);context=fixture(root)
            hashes={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}
            data=load_frame(context,63,mode='combined')
            np.testing.assert_allclose(data['bev_points'][:2],[[0,0],[.1,0]],atol=1e-7)
            np.testing.assert_allclose(data['bev_points'][2:],[[1,0],[.5,0],[-1,0],[0,0]])
            np.testing.assert_array_equal(data['bev_colors'][:3],[[0,255,255],[0,255,255],[65,135,240]])
            self.assertIn('1 个目标',data['status']);self.assertIn('测量坐标不变',data['status'])
            self.assertTrue(data['semantic_classes'])
            self.assertEqual(hashes,{p:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()})

    def test_combined_bev_draws_dashed_rectangles_only_for_optimized_support(self):
        from PySide6.QtGui import QPen
        with tempfile.TemporaryDirectory() as tmp:
            data=load_frame(fixture(Path(tmp)),63,mode='combined')
            canvas=PointCanvas(bev=True);canvas.data=data;styles=[]
            try:
                with patch('tools.lidar_annotator.live_preview.QPainter') as painter:
                    current=[]
                    painter.return_value.setPen.side_effect=lambda pen:current.append(QPen(pen))
                    painter.return_value.drawRect.side_effect=lambda rect:styles.append(current[-1].style())
                    canvas.paintEvent(None)
                    self.assertEqual(painter.return_value.drawRect.call_count,1)
                    self.assertEqual(styles,[Qt.PenStyle.DashLine])
                    rect=painter.return_value.drawRect.call_args.args[0]
                    scale,_,_=canvas.transform_values()
                    points=data['targets'][0]['optimized_points'];screen=np.c_[-points[:,1],-points[:,0]]
                    np.testing.assert_allclose([rect.left(),rect.top()],screen.min(0)-3/scale,atol=1e-6)
                    np.testing.assert_allclose([rect.right(),rect.bottom()],screen.max(0)+3/scale,atol=1e-6)
                # Without retained support, a saved centre alone must not invent a box.
                for target in data['targets']:target['optimized_points']=np.empty((0,3))
                with patch('tools.lidar_annotator.live_preview.QPainter') as painter:
                    canvas.paintEvent(None);painter.return_value.drawRect.assert_not_called()
            finally:canvas.close()

    def test_combined_fallback_labels_follow_original_pixels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);context=fixture(root);path=root/'raw/frames/000000.npz'
            with np.load(path) as z:arrays={k:z[k] for k in z.files if k!='point_raw_labels'}
            np.savez(path,**arrays)
            mask=np.zeros((8,8),np.uint8);mask[:,4]=1;mask[:,5]=3
            Image.fromarray(mask).save(root/'data/labels/zgc/image_masks/000000.png')
            data=load_frame(context,63,mode='combined')
            # Use the original pixel labels; no depth-based recoloring.
            self.assertLess(data['camera_points'][1,0],5)
            np.testing.assert_array_equal(data['bev_colors'][:2],[[0,255,255],[0,255,255]])

    def test_controls_auto_fit_dashed_bounds_and_preserve_view_on_stage_toggle(self):
        with tempfile.TemporaryDirectory() as tmp:
            context=fixture(Path(tmp));window=AnnotationWindow(DEFAULT)
            try:
                window.input_format.setCurrentIndex(window.input_format.findData('lmdb'));window.lmdb_output.setText(context['output']);from .test_qt import use_cached_work;use_cached_work(window,context['output'])
                window.track_button.click()
                deadline=time.monotonic()+5
                while window.frame_reader is not None and time.monotonic()<deadline:self.app.processEvents();time.sleep(.01)
                self.assertIsNotNone(window.track_data)
                self.assertFalse(window.track_controls.isHidden())
                window.select_target(7);self.assertGreater(window.bev_view.zoom,1)
                camera=window.camera_view;bev=window.bev_view
                saved=(bev.zoom,bev.pan.x(),bev.pan.y(),camera.zoom,camera.pan.x(),camera.pan.y())
                window.track_stage.setCurrentIndex(2)
                self.assertEqual(saved,(bev.zoom,bev.pan.x(),bev.pan.y(),camera.zoom,camera.pan.x(),camera.pan.y()))
                for canvas in (camera,bev):
                    with patch('tools.lidar_annotator.live_preview.QPainter') as painter:
                        canvas.paintEvent(None)
                        self.assertEqual(painter.return_value.drawRect.call_count,2)
                        # Copy mutable pens at call time to check their style.
                    from PySide6.QtGui import QPen
                    pens=[]
                    with patch('tools.lidar_annotator.live_preview.QPainter') as painter:
                        painter.return_value.setPen.side_effect=lambda pen:pens.append(QPen(pen))
                        canvas.paintEvent(None)
                    self.assertTrue(any(p.style()==Qt.PenStyle.DashLine for p in pens))
                window.target_id.setCurrentIndex(0);self.assertEqual(window.bev_view.zoom,1)
                window.combined_button.click()
                while window.frame_reader is not None and time.monotonic()<deadline:self.app.processEvents();time.sleep(.01)
                self.assertTrue(window.track_controls.isHidden());self.assertEqual(window.bev_view.data['mode'],'combined')
                self.assertEqual(window.bev_view.zoom,1)
            finally:window.close()

    def test_clicking_mask_selects_the_same_numeric_id(self):
        from PySide6.QtTest import QTest
        with tempfile.TemporaryDirectory() as tmp:
            data=track_view(load_frame(fixture(Path(tmp)),63,mode='tracks'))
            canvas=PointCanvas();canvas.set_frame(data);selected=[];canvas.target_selected.connect(selected.append)
            try:
                scale,w,h=canvas.transform_values()
                position=QPointF(canvas.width()/2,canvas.height()/2)+canvas.pan+QPointF((4.3-w/2)*scale,(4.3-h/2)*scale)
                QTest.mouseClick(canvas,Qt.MouseButton.LeftButton,pos=position.toPoint())
                self.assertEqual(selected,[7])
            finally:canvas.close()

    def test_frame_navigation_keeps_selected_id_and_reports_missing_id(self):
        import shutil
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);context=fixture(root)
            index=root/'raw/index.json';rows=json.loads(index.read_text())['rows']
            rows.append(dict(rows[0],path='frames/000001.npz',original_frame_id=64));index.write_text(json.dumps({'rows':rows}))
            shutil.copyfile(root/'raw/frames/000000.npz',root/'raw/frames/000001.npz')
            # Second frame has no instance masks, which must not select another ID.
            window=AnnotationWindow(DEFAULT)
            try:
                window.input_format.setCurrentIndex(window.input_format.findData('lmdb'));window.lmdb_output.setText(context['output']);from .test_qt import use_cached_work;use_cached_work(window,context['output']);window.track_button.click()
                deadline=time.monotonic()+5
                while window.frame_reader is not None and time.monotonic()<deadline:self.app.processEvents();time.sleep(.01)
                window.select_target(7);window.next_frame.click()
                while window.frame_reader is not None and time.monotonic()<deadline:self.app.processEvents();time.sleep(.01)
                self.assertEqual(window.preview_frame.value(),64)
                self.assertEqual(window.target_id.currentData(),7)
                self.assertIn('当前帧无此 ID',window.preview_status.text())
                self.assertEqual(window.camera_view.data['track_layers'],[])
            finally:window.close()

    def test_slider_wheel_up_previous_down_next_one_frame(self):
        slider=FrameSlider(Qt.Orientation.Horizontal);slider.setRange(0,9);slider.setValue(4)
        try:
            for delta,expected in ((120,3),(-120,4)):
                event=QWheelEvent(QPointF(5,5),QPointF(5,5),QPoint(),QPoint(0,delta),Qt.MouseButton.NoButton,Qt.KeyboardModifier.NoModifier,Qt.ScrollPhase.ScrollUpdate,False)
                self.app.sendEvent(slider,event);self.assertEqual(slider.value(),expected)
        finally:slider.close()

    def test_mask_updates_invalidate_preview(self):
        with tempfile.TemporaryDirectory() as tmp:
            context=fixture(Path(tmp));before=fingerprint(context,63)
            path=Path(tmp)/'data/visual_instances/000000.npz';path.write_bytes(path.read_bytes()+b'updated')
            self.assertNotEqual(fingerprint(context,63),before)

    def test_bev_grid_distance_labels_follow_zoom_pan_and_coordinate_signs(self):
        canvas=PointCanvas(bev=True);canvas.resize(640,640);canvas.data={'limits':(20,-3,6)}
        try:
            scale,_,_=canvas.transform_values();labels={text:(x,y) for x,y,text in canvas.grid_labels(scale)}
            self.assertEqual(labels['Y 0 m'][0],320)
            self.assertEqual(labels['X 0 m'][1],320)
            self.assertLess(labels['Y 5 m'][0],labels['Y 0 m'][0])
            self.assertLess(labels['X 5 m'][1],labels['X 0 m'][1])
            canvas.zoom=2;canvas.pan=QPointF(80,-40)
            scale,_,_=canvas.transform_values();labels={text:(x,y) for x,y,text in canvas.grid_labels(scale)}
            self.assertEqual(labels['Y 0 m'][0],400)
            self.assertEqual(labels['X 0 m'][1],280)
            self.assertAlmostEqual(labels['Y -5 m'][0]-labels['Y 0 m'][0],5*scale)
            with patch('tools.lidar_annotator.live_preview.QPainter') as painter:
                canvas.draw_grid_labels(painter.return_value,scale)
                self.assertEqual(painter.return_value.drawText.call_args_list[0].args[-1],'每格 5 m × 5 m')
            canvas.data['limits']=(10,-3,6)
            with patch('tools.lidar_annotator.live_preview.QPainter') as painter:
                canvas.draw_grid_labels(painter.return_value,scale)
                self.assertEqual(painter.return_value.drawText.call_args_list[0].args[-1],'每格 2.5 m × 2.5 m')
        finally:canvas.close()
