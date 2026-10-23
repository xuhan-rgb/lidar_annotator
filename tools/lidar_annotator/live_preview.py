"""Read current frame artifacts and draw independent, zoomable Qt views."""
import json
import html
from pathlib import Path
import numpy as np
from PIL import Image
from PySide6.QtCore import QPointF,QRectF,Qt,QThread,Signal
from PySide6.QtGui import QColor,QFont,QImage,QPainter,QPainterPath,QPen,QPolygonF
from PySide6.QtWidgets import QWidget
from .recipe_config import load_recipe
from .display_range import DEFAULT_LIMITS,point_mask
from .track_preview import load_targets,ignored_support_indices,project_points,vehicle_boxes_in_frame,vehicle_support_mask
from tools.autolabel.zgc_labeling.finalize_person_positions import load_positions


def semantic_legend_html(classes):
    names={'ground':'地面','person':'行人','vehicle':'车辆','building':'建筑','foliage':'树冠 / 植被','fence':'围栏 / 灌木','tree trunk':'树干','pole / wire':'立柱 / 电线','ground object':'地面障碍物','roof / eaves':'屋顶 / 屋檐'}
    entries=[]
    for item in classes:
        color='#{:02X}{:02X}{:02X}'.format(*item['rgb'])
        name=html.escape(names.get(item['name'],item['name'])).replace(' ','&nbsp;')
        entries.append(f'<span style="color:{color}">■</span>&nbsp;{name}')
    return '语义颜色： '+' &nbsp; '.join(entries) if entries else ''


def semantic_point_labels(arrays,image_labels,targets=(),vehicle_boxes=None):
    """The same semantic/optimized-support policy for single frames and maps."""
    if 'point_raw_labels' in arrays:
        result=np.asarray(arrays['point_raw_labels'],np.int32).copy()
    else:
        result=np.full(len(arrays['xyz']),255,np.int32)
        if image_labels is not None:
            uv=arrays['pixels_uv'];h,w=image_labels.shape
            valid=arrays.get('visible',arrays.get('in_image'))&np.isfinite(uv).all(1)&(arrays['camera_depth']>0)
            valid&=(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
            pixels=uv[valid].astype(int);result[valid]=image_labels[pixels[:,1],pixels[:,0]]
    result[~np.isin(result,range(10))]=255
    if vehicle_boxes is not None:
        support=vehicle_support_mask(arrays['xyz'],vehicle_boxes)
        result[(result==2)&~support]=255
        result[support&(result!=1)]=2
    rejected=ignored_support_indices(targets)
    result[rejected[result[rejected]==1]]=255
    return result.astype(np.uint8)


def confirmed_labels(paths,records):
    """Use saved, accepted YOLO-confirmed labels; raw SAM candidates stay intact."""
    people=[];vehicles=[]
    if paths['person_metadata'].is_file():
        people=json.loads(paths['person_metadata'].read_text()).get('instances',[])
    people={r['sam_source_instance_id']:r for r in people
                if r.get('status')=='accepted_xy_from_all_exclusive_mask_points'
                and r.get('person_class_confirmed') and (r.get('yolo_match') or {}).get('score',0)>=.4
                and len((r.get('yolo_match') or {}).get('bbox_xyxy',[]))==4}
    if paths['vehicle_metadata'].is_file():
        vehicles=json.loads(paths['vehicle_metadata'].read_text()).get('all_detections',[])
    vehicles=[r for r in vehicles if r.get('status')=='review_label'
              and (r.get('class_verification') or {}).get('yolo_score',0)>=.4
              and len((r.get('class_verification') or {}).get('yolo_bbox_xyxy',[]))==4]
    selected=[]
    for r in records:
        if r.get('prompt')=='person':
            person=people.get(r.get('source_instance_id',r.get('track_id')))
            if person:
                match=person['yolo_match']
                selected.append(dict(r,bbox_xyxy=match['bbox_xyxy'],score=match['score'],bbox_source='yolo26x'))
        else:
            matches=[v for v in vehicles if np.array_equal(r.get('bbox_xyxy'),v.get('sam_bbox_xyxy'))]
            if len(matches)==1:
                match=matches[0]['class_verification']
                selected.append(dict(r,bbox_xyxy=match['yolo_bbox_xyxy'],score=match['yolo_score'],bbox_source='yolo26x'))
    return selected,{int(v['prediction_index']) for v in vehicles}


def frame_inputs(context,frame,labels=True):
    out=Path(context['output']);raw=out/'raw';source=Path(context.get('backend',{}).get('source_index',out/'data/labels/zgc/index.json')).parent
    index=raw/'index.json' if (raw/'index.json').is_file() else source/'index.json'
    rows=json.loads(index.read_text())['rows']
    local,row=next(((i,r) for i,r in enumerate(rows) if int(r.get('original_frame_id',r.get('source_frame_id',Path(r['path']).stem)))==frame),(None,None))
    if row is None:raise ValueError(f'当前任务没有原始帧 {frame}')
    stem=f'{local:06d}' if index.parent==raw else Path(row['path']).stem
    data=source/'frames'/f'{stem}.npz'
    if not labels or not data.is_file():data=index.parent/row['path']
    if index.parent!=raw:local=int(row.get('local_frame_id',frame))
    temporal=Path(context.get('backend',{}).get('temporal_output',out/'temporal'))
    tracks=temporal/'map_review/sam3_full_video/interior_half_review/consistent_tracks/centers_only.npz'
    return dict(out=out,index=index,row=row,local=local,stem=stem,data=data,image=Path(row['image_path']) if 'image_path' in row else Path(row['original_shard'])/'data.mdb',
                mask=source/'image_masks'/f'{stem}.png',records=out/'data/visual_instances'/f'{stem}.json',
                recipe=out/'recipe.json',tracks=tracks,poses=temporal/'poses.npz',
                person_metadata=Path(context.get('backend',{}).get('person_root',out/'data/person'))/'metadata'/f'{stem}.json',
                vehicle_metadata=Path(context.get('backend',{}).get('vehicle_root',out/'data/vehicle'))/'refined'/f'{stem}.json',
                vehicle_labels=Path(context.get('backend',{}).get('vehicle_root',out/'data/vehicle'))/'refined'/f'{stem}.npz',
                instance_masks=out/'data/visual_instances'/f'{stem}.npz',track_masks=temporal/'map_review/sam3_full_video/masks'/f'{stem}.npz',
                cars=temporal/'map_review/history_fit/bev_dedup/vehicle_map_boxes.npz',id_map=temporal/'map_review/sam3_full_video/native_to_display_ids.json',tracks_report=tracks.parent/'report.json')


def fingerprint(context,frame,labels=True):
    paths=frame_inputs(context,frame,labels)
    names=('index','data','image','mask','records','recipe','tracks','poses','cars','id_map','instance_masks','track_masks','tracks_report','person_metadata','vehicle_metadata','vehicle_labels') if labels else ('index','data','image')
    files=[paths[k] for k in names]
    if labels:files+=[paths['tracks'].with_name('final_centers.npz')]+sorted((paths['tracks'].parent/'metadata').glob(f"{paths['local']:06d}_I*.json"))
    return tuple((str(path),path.stat().st_mtime_ns,path.stat().st_size) if path.is_file() else (str(path),None) for path in files)


def load_frame(context,frame,labels=True,limits=DEFAULT_LIMITS,mode='combined'):
    from matplotlib import colormaps
    if mode=='projection':labels=False
    p=frame_inputs(context,frame,labels)
    with np.load(p['data'],allow_pickle=False) as archive:a={k:archive[k] for k in archive.files}
    if 'image_path' in p['row']:
        with Image.open(p['image']) as source:image=source.convert('RGB')
    else:
        import hashlib,lmdb
        from .lmdb_input import decode_frame
        row=p['row']
        with lmdb.open(row['original_shard'],readonly=True,lock=False,create=False) as env:
            with env.begin() as txn:payload=txn.get(row['original_lmdb_key'].encode())
        if payload is None:raise ValueError('当前帧的原始图像记录不存在')
        if row.get('original_payload_sha256') and hashlib.sha256(payload).hexdigest()!=row['original_payload_sha256']:raise ValueError('当前帧原始记录校验失败')
        _,_,image,_=decode_frame(payload)
    filtered=point_mask(a['xyz'],limits);extent,zmin,zmax=limits
    visible=a['visible'] if 'visible' in a else a['in_image']
    semantic=False;ids=None;records=[];centres=np.empty((0,2));boxes=[];box_ids=[];centre_ids=[];camera_lines=[];camera_centres=[];optimized_cars=None
    centre_track_only=np.empty(0,bool);camera_centre_track_only=np.empty(0,bool);person_position_policy=''
    if labels and mode!='tracks' and p['mask'].is_file():
        with Image.open(p['mask']) as mask:ids=np.asarray(mask)
        if ids.shape!=(image.height,image.width):raise ValueError('语义 mask 与原图尺寸不一致')
        recipe=load_recipe(p['recipe'] if p['recipe'].is_file() else None)
        rgb=np.asarray(image).copy();palette=np.full((256,3),100,np.uint8)
        for item in recipe['raw_classes']:palette[item['id']]=item['rgb']
        valid=np.isin(ids,[item['id'] for item in recipe['raw_classes']])
        rgb[valid]=(rgb[valid]*.7+palette[ids[valid]]*.3).astype(np.uint8)
        if mode=='semantic':rgb[:]=0;rgb[valid]=palette[ids[valid]]
        image=Image.fromarray(rgb);semantic_palette=palette.copy();semantic=True
    if labels and p['records'].is_file():
        records=json.loads(p['records'].read_text()).get('records',[])
        mapping=json.loads(p['id_map'].read_text()) if p['id_map'].is_file() else {}
        for record in records:
            tid=record.get('track_id')
            record['display_id']=mapping.get(str(tid),tid)
    confirmed,vehicle_seeds=confirmed_labels(p,records) if labels and mode=='combined' else (records,set())
    person_ids={r['display_id'] for r in confirmed if r.get('prompt')=='person'}
    if labels and mode!='semantic' and p['tracks'].is_file():
        tracks=load_positions(p['tracks']);person_position_policy=tracks['position_policy']
        chosen=(tracks['frame_ids']==p['local'])&tracks['position_available']
        if mode=='combined':chosen&=np.isin(tracks['sam_video_ids'],list(person_ids))
        points=tracks['corrected_point_lidar'][chosen];inside=point_mask(points,limits);centre_ids=tracks['sam_video_ids'][chosen][inside];points=points[inside];centres=points[:,:2]
        centre_track_only=tracks['track_only'][chosen][inside]
        camera=points@a['cam_extrinsic'][:3,:3].T+a['cam_extrinsic'][:3,3];positive=camera[:,2]>.1;pixels=camera[positive]@a['cam_intrinsic'].T
        camera_centres=pixels[:,:2]/pixels[:,2:3];camera_centre_track_only=centre_track_only[positive]
    final_cars=labels and mode!='semantic' and p['cars'].is_file() and p['poses'].is_file()
    if final_cars or (labels and mode=='combined' and p['vehicle_labels'].is_file()):
        if final_cars:
            with np.load(p['poses'],allow_pickle=False) as poses:pose=poses['transforms'][p['local']]
            with np.load(p['cars'],allow_pickle=False) as cars:
                world_boxes=cars['box_3d_map'].reshape(-1,7);car_ids=cars['track_ids']
                keep=np.isfinite(world_boxes).all(1)&(world_boxes[:,3:6]>0).all(1)
                car_boxes=vehicle_boxes_in_frame(world_boxes[keep],pose);car_ids=car_ids[keep]
            optimized_cars=car_boxes
        else:
            with np.load(p['vehicle_labels'],allow_pickle=False) as cars:
                keep=np.isin(cars['seed_ids'],list(vehicle_seeds));car_boxes=cars['box_3d'][keep].reshape(-1,7).copy();car_ids=cars['seed_ids'][keep]
        for box,tid in zip(car_boxes,car_ids):
            xy=np.array([[-1,-1],[1,-1],[1,1],[-1,1]],float)*box[3:5]/2
            co,si=np.cos(box[6]),np.sin(box[6]);xy=xy@np.array([[co,si],[-si,co]])+box[:2]
            if (xy.max(0)<-extent).any() or (xy.min(0)>extent).any() or box[2]+box[5]/2<zmin or box[2]-box[5]/2>zmax:continue
            boxes.append(xy);box_ids.append(int(tid))
            # Keep the established near-plane/image clipping for 3D box edges.
            from tools.autolabel.wilddet3d_zgc.projection_clipping import project_clipped_box_edges
            corners=np.concatenate([np.c_[xy,np.full(4,box[2]-box[5]/2)],np.c_[xy,np.full(4,box[2]+box[5]/2)]])
            camera=corners@a['cam_extrinsic'][:3,:3].T+a['cam_extrinsic'][:3,3]
            edges=((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7))
            # Require current visible scan support before displaying a map box in RGB.
            support=filtered&vehicle_support_mask(a['xyz'],[box])&visible
            if support.sum()>=3:camera_lines.extend((u,v) for _,u,v in project_clipped_box_edges(camera,edges,a['cam_intrinsic'],image.width,image.height))
    targets=load_targets(p,a,records,limits) if labels and mode in ('tracks','combined','semantic') else []
    semantic_targets=targets
    if mode=='combined':
        records=confirmed
        keys={(r.get('display_id'),r.get('prompt')) for r in records}
        targets=[t for t in targets if (t['id'],t['prompt']) in keys]
    if mode=='semantic':records=[];targets=[]
    xyz=a['xyz'];depth=a['camera_depth'];uv=a['pixels_uv']

    bev=filtered
    projected=filtered&visible&np.isfinite(uv).all(1)&np.isfinite(depth)&(depth>0)
    cmap=colormaps['turbo_r']
    palette=(cmap(np.linspace(0,1,32))[:,:3]*255).astype(np.uint8)
    def colors(values):return palette[(np.clip(values/40,0,1)*31).astype(np.int32)]
    camera_colors=colors(depth[projected]);bev_colors=colors(np.linalg.norm(xyz[bev],axis=1))
    if labels and mode=='combined' and not semantic and ('point_raw_labels' in a or optimized_cars is not None):
        recipe=load_recipe(p['recipe'] if p['recipe'].is_file() else None)
        semantic_palette=np.full((256,3),100,np.uint8)
        for item in recipe['raw_classes']:semantic_palette[item['id']]=item['rgb']
        semantic=True
    if mode in ('semantic','combined') and semantic:
        ids3d=semantic_point_labels(a,ids,semantic_targets,optimized_cars if mode=='combined' else None)
        if ids3d is not None:
            valid_ids=np.asarray(ids3d,dtype=np.int32);valid_ids=np.where((valid_ids>=0)&(valid_ids<256),valid_ids,255);bev_colors=semantic_palette[valid_ids[bev]]
            if mode=='combined':camera_colors=semantic_palette[valid_ids[projected]]
    if mode=='semantic':projected=np.zeros(len(xyz),bool);camera_colors=np.empty((0,3),np.uint8)
    coverage={}
    if mode=='projection':
        from .camera_coverage import camera_coverage
        coverage=camera_coverage(a['cam_intrinsic'],a['cam_extrinsic'],image.size,xyz,limits)
    image_bytes=image.tobytes();qimage=QImage(image_bytes,image.width,image.height,image.width*3,QImage.Format.Format_RGB888).copy()
    optimization_note=''
    if mode=='combined':
        optimized_count=sum(t['corrected'] is not None for t in semantic_targets);ignored_count=len(ignored_support_indices(semantic_targets))
        reviews=sum(bool((t.get('semantic_selection') or {}).get('review_required')) and t.get('person_class_confirmed') is not False for t in semantic_targets)
        optimization_note=f' · 语义点云：{optimized_count} 个目标使用已校验支持点，{ignored_count} 个原行人点语义忽略；测量坐标不变' if semantic else ' · 语义标签尚未生成'
        if reviews:optimization_note+=f' · {reviews} 个点团仍待核查'
        unconfirmed=sum(t.get('person_class_confirmed') is False for t in semantic_targets)
        if unconfirmed:optimization_note+=f' · {unconfirmed} 个类别未确认候选，行人语义已标忽略'
        optimization_note+=' · 车辆框与支持点使用时序优化结果' if optimized_cars is not None else ' · 车辆时序优化结果尚未生成，显示单帧标签'
        optimization_note+=' · 图像检测框仅显示 YOLO26x 验证且已接受的标签'
        if person_position_policy:optimization_note+=' · 行人位置：'+('全局参考轨迹' if person_position_policy=='global_reference_xy_v1' else '旧射线结果，执行流程后更新')
        if centre_track_only.any():optimization_note+=f' · {int(centre_track_only.sum())} 个位置仅为轨迹估计'
    return dict(frame=frame,image=qimage,palette=np.concatenate([palette,semantic_palette]) if mode in ('semantic','combined') and semantic else palette,limits=limits,mode=mode,camera_points=uv[projected],camera_colors=camera_colors,
                bev_points=xyz[bev,:2],bev_colors=bev_colors,camera_point_indices=np.flatnonzero(projected),bev_point_indices=np.flatnonzero(bev),
                records=records,centres=centres,centre_ids=centre_ids,boxes=boxes,box_ids=box_ids,camera_lines=camera_lines,camera_centres=camera_centres,semantic=semantic,
                centre_track_only=centre_track_only,camera_centre_track_only=camera_centre_track_only,person_position_policy=person_position_policy,
                semantic_classes=recipe['raw_classes'] if mode in ('semantic','combined') and semantic else [],coverage=coverage,
                targets=targets,track_rgb=np.asarray(image).copy() if mode=='tracks' else None,extrinsic=a['cam_extrinsic'],intrinsic=a['cam_intrinsic'],
                status=f'帧 {frame} · XY ±{extent:g} m / Z [{zmin:g}, {zmax:g}] m · 投影点 {int(projected.sum())} · BEV 点 {int(bev.sum())} · '+('原始深度投影 · 相机几何范围不代表完整标注或负样本' if mode=='projection' else '原始深度投影' if mode=='projection' or not labels else '语义 mask' if mode=='semantic' and semantic else '最新语义标签' if semantic else '跟踪 ID' if mode=='tracks' and any(r.get('track_id') is not None for r in records) else '跟踪 ID 尚未生成' if mode=='tracks' else '原始数据（语义标签尚未生成）')+optimization_note)


class FrameReader(QThread):
    loaded=Signal(object,object)
    failed=Signal(str)

    def __init__(self,context,frame,labels,key,parent=None,limits=DEFAULT_LIMITS,mode='combined'):
        super().__init__(parent);self.context=context;self.frame=frame;self.labels=labels;self.key=key;self.limits=limits;self.mode=mode

    def run(self):
        try:self.loaded.emit(self.key,load_frame(self.context,self.frame,self.labels,self.limits,self.mode))
        except Exception as exc:self.failed.emit(str(exc))


class PointCanvas(QWidget):
    """Paint points at viewport resolution; zoom/pan does not rescale a screenshot."""
    target_selected=Signal(object)

    def __init__(self,bev=False,parent=None):
        super().__init__(parent);self.bev=bev;self.show_coverage=True;self.data=None;self.point_size=3;self.zoom=1.;self.pan=QPointF();self.drag=None;self.groups=[]
        self.setMinimumSize(160,180);self.setMouseTracking(True)

    def set_frame(self,data):
        self.data=data;points=data['bev_points' if self.bev else 'camera_points'];colors=data['bev_colors' if self.bev else 'camera_colors']
        self.coverage_path=QPainterPath()
        if self.bev and data.get('coverage'):
            from .camera_coverage import boundary_curves
            for curve in boundary_curves(data['coverage']['boundary_xy']):
                self.coverage_path.moveTo(-float(curve[0,1]),-float(curve[0,0]))
                for x,y in curve[1:]:self.coverage_path.lineTo(-float(y),-float(x))
        if self.bev:points=np.c_[-points[:,1],-points[:,0]]
        # A fixed number of batched draws, with no sorting or pairwise operations.
        self.groups=[]
        for color in dict.fromkeys(tuple(map(int,c)) for c in data['palette']):
            selected=(colors==color).all(1)
            if selected.any():self.groups.append((QColor(*map(int,color)),QPolygonF([QPointF(float(x),float(y)) for x,y in points[selected]])))
        self.update()

    def reset_view(self):self.zoom=1.;self.pan=QPointF();self.update()

    def transform_values(self):
        w,h=(2*self.data['limits'][0],)*2 if self.bev else (self.data['image'].width(),self.data['image'].height())
        return min(self.width()/w,self.height()/h)*self.zoom,w,h

    def paintEvent(self,event):
        painter=QPainter(self);painter.fillRect(self.rect(),QColor('#14171c'))
        if self.data is None:painter.setPen(Qt.GlobalColor.white);painter.drawText(self.rect(),Qt.AlignmentFlag.AlignCenter,'选择帧号，实时读取数据');return
        scale,w,h=self.transform_values();painter.translate(self.width()/2+self.pan.x(),self.height()/2+self.pan.y());painter.scale(scale,scale)
        if self.bev:
            painter.setPen(QPen(QColor('#343a40'),0))
            extent=self.data['limits'][0]
            for v in np.linspace(-extent,extent,9):painter.drawLine(QPointF(float(v),-extent),QPointF(float(v),extent));painter.drawLine(QPointF(-extent,float(v)),QPointF(extent,float(v)))
        else:painter.translate(-w/2,-h/2);painter.drawImage(QPointF(0,0),self.data['image'])
        for color,polygon in self.groups:
            pen=QPen(color,self.point_size);pen.setCosmetic(True);pen.setCapStyle(Qt.PenCapStyle.RoundCap);painter.setPen(pen);painter.drawPoints(polygon)
        if self.bev and self.data['mode']=='projection' and self.show_coverage:
            pen=QPen(QColor('#d980fa'),2,Qt.PenStyle.DashLine);pen.setCosmetic(True);painter.setPen(pen);painter.drawPath(self.coverage_path)
        pen=QPen(QColor('#00ffff'),2);pen.setCosmetic(True);painter.setPen(pen)
        font=QFont();font.setPixelSize(max(1,round(14/scale)));painter.setFont(font)
        if self.bev:
            if self.data['mode']=='combined':
                pen.setStyle(Qt.PenStyle.DashLine);painter.setPen(pen)
                for target in self.data.get('targets',[]):
                    points=target['optimized_points']
                    if target['corrected'] is None or not len(points):continue
                    screen=np.c_[-points[:,1],-points[:,0]]
                    lo=screen.min(0)-3/scale;hi=screen.max(0)+3/scale
                    painter.drawRect(QRectF(float(lo[0]),float(lo[1]),float(hi[0]-lo[0]),float(hi[1]-lo[1])))
                pen.setStyle(Qt.PenStyle.SolidLine);painter.setPen(pen)
            radius=5/scale
            flags=self.data.get('centre_track_only',np.zeros(len(self.data['centres']),bool))
            for i,((x,y),tid) in enumerate(zip(self.data['centres'],self.data['centre_ids'])):
                pen.setColor(QColor('#ffbf66' if flags[i] else '#00ffff'));painter.setPen(pen)
                if flags[i]:
                    painter.drawLine(QPointF(-y-radius,-x-radius),QPointF(-y+radius,-x+radius));painter.drawLine(QPointF(-y-radius,-x+radius),QPointF(-y+radius,-x-radius))
                else:painter.drawEllipse(QPointF(-y,-x),radius,radius)
                painter.drawText(QPointF(-y+radius,-x),str(int(tid)))
            pen.setColor(QColor('#ff9b1e'));painter.setPen(pen)
            for box,tid in zip(self.data['boxes'],self.data['box_ids']):
                painter.drawPolygon(QPolygonF([QPointF(-float(y),-float(x)) for x,y in box]));x,y=box.mean(0);painter.drawText(QPointF(-float(y),-float(x)),str(tid))
        else:
            radius=5/scale
            flags=self.data.get('camera_centre_track_only',np.zeros(len(self.data['camera_centres']),bool))
            for i,(x,y) in enumerate(self.data['camera_centres']):
                pen.setColor(QColor('#ffbf66' if flags[i] else '#00ffff'));painter.setPen(pen)
                if flags[i]:
                    painter.drawLine(QPointF(float(x)-radius,float(y)-radius),QPointF(float(x)+radius,float(y)+radius));painter.drawLine(QPointF(float(x)-radius,float(y)+radius),QPointF(float(x)+radius,float(y)-radius))
                else:painter.drawEllipse(QPointF(float(x),float(y)),radius,radius)
            pen.setColor(QColor('#ff9b1e'));painter.setPen(pen)
            for a,b in self.data['camera_lines']:painter.drawLine(QPointF(float(a[0]),float(a[1])),QPointF(float(b[0]),float(b[1])))
            for item in self.data['records']:
                bbox=item.get('bbox_xyxy')
                if bbox:
                    pen.setColor(QColor('#00ffff' if item.get('prompt')=='person' else '#ff9b1e'));painter.setPen(pen)
                    x1,y1,x2,y2=bbox;painter.drawRect(QRectF(x1,y1,x2-x1,y2-y1))
                    tid=item.get('display_id')
                    if tid is not None:
                        font=QFont();font.setPixelSize(max(1,round(14/scale)));painter.setFont(font);painter.drawText(QPointF(x1,max(0,y1-4/scale)),str(tid))
        if self.data['mode']=='tracks':self.draw_track_layers(painter,scale)
        painter.resetTransform();painter.setPen(Qt.GlobalColor.white)
        if self.bev:self.draw_grid_labels(painter,scale)
        painter.setFont(self.font());painter.drawText(8,18,f"BEV：XY ±{self.data['limits'][0]:g} m；X 向上，Y 向左" if self.bev else '语义 mask：类别配色' if self.data['mode']=='semantic' else '实例 mask / 目标点云' if self.data['mode']=='tracks' else '联合标注：车辆框 / 语义点云' if self.data['mode']=='combined' and self.data['semantic'] else '相机投影：Z 深度 0–40 m')
        painter.drawText(8,self.height()-10,'红色近 → 蓝色远 · 紫色虚线：相机几何范围' if self.data['mode']=='projection' and self.bev and self.show_coverage else '类别配色 · 滚轮缩放 · 左键拖动' if self.data['mode']=='semantic' or (self.data['mode']=='combined' and self.data['semantic']) else '灰色点：背景点云 · 虚线框为目标点集范围 · 滚轮缩放 · 左键拖动' if self.data['mode']=='tracks' else '红色近 → 蓝色远 · 滚轮缩放 · 左键拖动')

    def grid_labels(self,scale):
        """Screen-edge coordinates for the existing grid: screen=(-Y,-X)."""
        extent=self.data['limits'][0];origin=QPointF(self.width()/2,self.height()/2)+self.pan
        labels=[];last_x=last_y=-float('inf')
        for value in np.linspace(-extent,extent,9):
            coordinate=0. if abs(value)<1e-10 else -float(value)
            x=origin.x()+value*scale;y=origin.y()+value*scale
            if 40<=x<=self.width()-40 and x-last_x>=68:
                labels.append((float(x),float(self.height()-30),f'Y {coordinate:g} m'));last_x=x
            if 54<=y<=self.height()-48 and y-last_y>=22:
                labels.append((8.,float(y),f'X {coordinate:g} m'));last_y=y
        return labels

    def draw_grid_labels(self,painter,scale):
        painter.setFont(self.font());painter.setPen(QColor('#d5dce3'))
        extent=self.data['limits'][0];step=extent/4
        painter.drawText(8,36,f'每格 {step:g} m × {step:g} m')
        for x,y,text in self.grid_labels(scale):
            if text.startswith('Y '):
                painter.drawText(QRectF(x-34,y-13,68,18),Qt.AlignmentFlag.AlignCenter,text)
            else:painter.drawText(QPointF(x,y),text)
        painter.setPen(Qt.GlobalColor.white)

    def fit_target(self,bounds):
        if bounds is None:return
        lower,upper=np.asarray(bounds,dtype=float)
        if self.bev:
            lower,upper=-upper[::-1],-lower[::-1]
        scale,w,h=self.transform_values();base=scale/self.zoom
        span=np.maximum(upper-lower,.8 if self.bev else 40.)
        self.zoom=max(.25,min(128.,min(self.width()/(span[0]*1.5),self.height()/(span[1]*1.5))/base))
        center=(lower+upper)/2
        if not self.bev:center-=np.array([w,h])/2
        self.pan=QPointF(*(-center*base*self.zoom));self.update()

    def draw_track_layers(self,painter,scale):
        def display(points):
            points=np.asarray(points).reshape(-1,2)
            return np.c_[-points[:,1],-points[:,0]] if self.bev else points
        def project_center(center):
            if center is None:return None
            if self.bev:return display(np.asarray(center[:2]).reshape(1,2))[0]
            uv,valid=project_points(np.asarray(center).reshape(1,3),self.data['extrinsic'],self.data['intrinsic'],(self.data['image'].height(),self.data['image'].width()))
            return uv[0] if valid[0] else None
        for layer in self.data.get('track_layers',[]):
            points=display(layer['bev' if self.bev else 'camera'])
            color=QColor(*layer['color']);pen=QPen(color,self.point_size+1);pen.setCosmetic(True);painter.setPen(pen)
            painter.drawPoints(QPolygonF([QPointF(float(x),float(y)) for x,y in points]))
            pen.setWidth(2);pen.setStyle(Qt.PenStyle.DashLine);painter.setPen(pen)
            if len(points):
                lo=points.min(0)-3/scale;hi=points.max(0)+3/scale
                painter.drawRect(QRectF(float(lo[0]),float(lo[1]),float(hi[0]-lo[0]),float(hi[1]-lo[1])))
                painter.drawText(QPointF(float(lo[0]),float(lo[1]-4/scale)),str(layer['id']))
            center=project_center(layer['center'])
            if center is not None:
                pen.setStyle(Qt.PenStyle.SolidLine);painter.setPen(pen);painter.drawEllipse(QPointF(*center),5/scale,5/scale)
        pen=QPen(QColor('#ffffff'),2);pen.setCosmetic(True);painter.setPen(pen)
        for before,after in self.data.get('track_arrows',[]):
            a=project_center(before);b=project_center(after)
            if a is None or b is None:continue
            painter.drawLine(QPointF(*a),QPointF(*b))
            delta=b-a;length=np.linalg.norm(delta)
            if length>1e-8:
                direction=delta/length;normal=np.array([-direction[1],direction[0]])
                for sign in (-1,1):
                    tail=b-direction*8/scale+normal*sign*4/scale
                    painter.drawLine(QPointF(*tail),QPointF(*b))

    def wheelEvent(self,event):
        factor=1.25 if event.angleDelta().y()>0 else .8;new=max(.25,min(128.,self.zoom*factor));factor=new/self.zoom
        anchor=event.position()-QPointF(self.width()/2,self.height()/2);self.pan=anchor-(anchor-self.pan)*factor;self.zoom=new;self.update();event.accept()

    def mousePressEvent(self,event):
        if event.button()==Qt.MouseButton.LeftButton:self.drag=event.position();self.press_position=event.position()

    def mouseMoveEvent(self,event):
        if self.drag is not None:self.pan+=event.position()-self.drag;self.drag=event.position();self.update()

    def mouseReleaseEvent(self,event):
        self.drag=None
        if self.data is None or self.data['mode']!='tracks' or not hasattr(self,'press_position'):return
        if (event.position()-self.press_position).manhattanLength()>5:return
        scale,w,h=self.transform_values()
        point=(event.position()-QPointF(self.width()/2,self.height()/2)-self.pan)/scale
        if not self.bev:
            x=int(point.x()+w/2);y=int(point.y()+h/2)
            hits=[t['id'] for t in self.data.get('targets',[]) if 0<=y<t['mask'].shape[0] and 0<=x<t['mask'].shape[1] and t['mask'][y,x]]
            if len(hits)==1:self.target_selected.emit(hits[0])
        else:
            candidates=[]
            for layer in self.data.get('track_layers',[]):
                cloud=layer['bev']
                if len(cloud):
                    distance=np.linalg.norm(np.c_[-cloud[:,1],-cloud[:,0]]-[point.x(),point.y()],axis=1).min()*scale
                    candidates.append((distance,layer['id']))
            if candidates:
                distance,tid=min(candidates)
                if distance<=10:self.target_selected.emit(tid)

