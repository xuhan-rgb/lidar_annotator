"""Publish portable per-frame labels beside KITTI raw data, preserving reviewed edits."""
import argparse
import hashlib
import io
import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image

from .kitti_input import frame_records
from .recipe_config import class_prompts,load_recipe

SCHEMA='lidar_dataset_labels_v1'


def json_bytes(value):return (json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8')
def digest(value):return hashlib.sha256(value).hexdigest()
def read_json(path):return json.loads(Path(path).read_text())
def signature(value):return digest(json_bytes({k:v for k,v in value.items() if k!='export_signature'}))


def publish(directory,stem,content,artifacts,backup):
    """JSON commits each frame bundle last; changed/reviewed bundles are protected."""
    directory=Path(directory);path=directory/f'{stem}.json'
    content={**content,'schema':SCHEMA,'review_status':'unreviewed','manual_ground_truth':False,
             'artifact_sha256':{name:digest(data) for name,data in artifacts.items()}}
    content['export_signature']=signature(content)
    files={**artifacts,path.name:json_bytes(content)}
    existing=[directory/name for name in files if (directory/name).exists()]
    if existing:
        try:
            old=read_json(path)
            if old.get('source_camera')!=content.get('source_camera'):return 'protected_other_camera'
            if old.get('schema')!=SCHEMA or old.get('review_status')!='unreviewed' or old.get('manual_ground_truth') or old.get('export_signature')!=signature(old):return 'protected_edited'
            for name,sha in old.get('artifact_sha256',{}).items():
                if Path(name).name!=name or not (directory/name).is_file() or digest((directory/name).read_bytes())!=sha:return 'protected_edited'
            if set(old.get('artifact_sha256',{}))-set(artifacts):return 'protected_incomplete'
            if all((directory/name).is_file() and (directory/name).read_bytes()==data for name,data in files.items()):return 'reused'
        except (OSError,ValueError,TypeError):return 'protected_edited'
        previous=backup/directory.name/stem;previous.mkdir(parents=True,exist_ok=True)
        for name in set(files)|set(old.get('artifact_sha256',{})):
            if (directory/name).is_file():shutil.copy2(directory/name,previous/name)
    directory.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.labels-',dir=directory) as tmp:
        for name,data in files.items():(Path(tmp)/name).write_bytes(data)
        for name in files:(Path(tmp)/name).replace(directory/name)
    return 'written'


def _image_bundle(job,row,recipe,camera,frame,source_frame):
    mask=Path(row['semantic_mask']);conf=Path(row['semantic_confidence']);instances=Path(row['visual_instances']);records=instances.with_suffix('.json')
    local=f"{row['local_frame']:06d}";yolo=job/'data/yolo/metadata'/f'{local}.json'
    available=dict(semantic=mask.is_file(),instances=instances.is_file() and records.is_file(),boxes_2d=yolo.is_file())
    if not any(available.values()):return None
    content=dict(frame_id=frame,source_frame_id=source_frame,source_camera=camera,available=available,instances=[],detections=[],negative_supervision=False)
    artifacts={};arrays={}
    if available['semantic']:
        artifacts['.png']=mask.read_bytes()
        if conf.is_file():
            with np.load(conf,allow_pickle=False) as z:arrays['semantic_confidence']=z['confidence']
    if available['instances']:
        with np.load(instances,allow_pickle=False) as z:arrays.update({k:z[k] for k in z.files})
        arrays['mask_bitorder']=np.asarray('little')
        raw=read_json(records)['records']
        if len(raw)!=len(arrays['masks_packed']):raise ValueError('实例 mask 与 2D 记录数量不一致')
        person_path=Path(row['person_metadata']);people=read_json(person_path).get('instances',[]) if person_path.is_file() else []
        people={str(p['sam_source_instance_id']):p for p in people}
        id_path=job/'temporal/map_review/sam3_full_video/native_to_display_ids.json';ids=read_json(id_path) if id_path.is_file() else {}
        prompt_classes={p:cid for cid in range(10) for p in class_prompts(recipe,cid)}
        for n,record in enumerate(raw):
            item=dict(record,mask_index=n,class_id=prompt_classes.get(record['prompt']),status='unconfirmed')
            native=record.get('track_id');item['native_track_id']=str(native) if native is not None else None
            item['track_id']=ids.get(str(native)) if record['prompt']=='person' else native
            person=people.get(str(record.get('source_instance_id')))
            if person is not None:item.update(status=person['status'],reasons=person.get('reasons',[]))
            content['instances'].append(item)
    if arrays:
        buf=io.BytesIO();np.savez_compressed(buf,**arrays);artifacts['.npz']=buf.getvalue()
    if yolo.is_file():
        for detection in read_json(yolo)['detections']:
            item=dict(detection,detector_class_id=detection['class_id'],status='candidate')
            item['class_id']=1 if detection['name']=='person' else 2 if detection['name'] in ('car','bus','truck') else None
            content['detections'].append(item)
    return content,artifacts


def _lidar_bundle(row,camera,frame,source_frame,points_path):
    with np.load(row['label_path'],allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
    semantic='point_raw_labels' in arrays
    objects='person_xyz' in arrays or 'vehicle_box_3d' in arrays
    if not semantic and not objects:return None
    raw=np.fromfile(points_path,dtype='<f4').reshape(-1,4)
    if not np.array_equal(raw[:,:3],arrays['xyz'].astype(np.float32),equal_nan=True):raise ValueError(f'点云标签与原始点顺序不一致：{points_path}')
    content=dict(frame_id=frame,source_frame_id=source_frame,source_camera=camera,coordinate_system='lidar',units='metres',
                 available=dict(semantic=semantic,objects=objects),point_count=len(raw),persons=[],vehicles=[],negative_supervision=False)
    artifacts={}
    if semantic:
        labels=np.asarray(arrays['point_raw_labels']).copy()
        if labels.shape!=(len(raw),):raise ValueError('点云标签数量与原始点云不一致')
        if 'person_semantic_ignored_indices' in arrays and 'person_semantic_filter_json' in arrays:
            audit=json.loads(str(np.asarray(arrays['person_semantic_filter_json']).item()))
            content['person_semantic_filter']=audit
        labels[~np.isin(labels,[*range(10),255])]=255
        if 'training_roi_mask' in arrays:labels[~arrays['training_roi_mask']]=255
        artifacts['.label']=labels.astype('<u4').tobytes()
    if 'person_xyz' in arrays:
        for i,(xyz,track,source) in enumerate(zip(arrays['person_xyz'],arrays['person_track_ids'],arrays['person_position_source'])):
            item=dict(class_id=1,track_id=int(track),position_xyz=xyz.tolist(),position_source=str(source),supervision='xy',box_3d_valid=False)
            if 'person_track_only' in arrays:item['track_only']=bool(arrays['person_track_only'][i])
            if 'person_support_counts' in arrays:item['current_support_points']=int(arrays['person_support_counts'][i])
            if 'person_position_policy' in arrays:item['position_policy']=str(arrays['person_position_policy'].item())
            content['persons'].append(item)
    if 'vehicle_box_3d' in arrays:
        for box,track in zip(arrays['vehicle_box_3d'],arrays['vehicle_track_ids']):content['vehicles'].append(dict(class_id=2,track_id=int(track),box_3d=box.tolist()))
    return content,artifacts


def export_labels(job,output=None):
    job=Path(job).expanduser().resolve();owner=read_json(job/'job.json');catalog=read_json(job/'annotations.json')
    sequence=Path(owner['source']) if owner.get('input_format')=='kitti' else Path(owner.get('dataset_output',job/'kitti'))
    camera=owner.get('camera','image_2');sequence,records=frame_records(sequence,camera)
    mapping={int(r['global_frame_id']):r for r in records};root=Path(output).expanduser().resolve() if output else sequence/'labels'
    if root==sequence or root in sequence.parents:raise ValueError('标签输出不能覆盖原始序列目录')
    recipe=load_recipe(job/'recipe.json');classes=dict(schema=SCHEMA,classes=recipe['raw_classes'],unknown_id=255,point_label_dtype='<u4',
        point_label_encoding='one raw semantic class ID per original point; no packed instance ID',box_3d_fields=['x','y','z','length','width','height','yaw'],yaw_unit='radians',coordinate_system='lidar')
    class_path=root/'classes.json';classes_data=json_bytes(classes)
    if class_path.exists() and class_path.read_bytes()!=classes_data:raise ValueError(f'已有类别定义不同，保留原文件：{class_path}')
    backup=job/'label_versions'/(datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid4().hex[:8])
    report=dict(status='complete',output=str(root),camera=camera,frames=[],manual_ground_truth=False)
    for row in catalog['rows']:
        original=int(row['frame_id']);record=mapping.get(original)
        if record is None:raise ValueError(f'原始序列中没有标签对应的帧：{original}')
        frame=int(record['local_frame_id']);image_stem=record['image_path'].stem;lidar_stem=record['points_path'].stem
        label_path=Path(row['label']);label_path=label_path if label_path.is_absolute() else job/label_path
        bundles=[(root/camera,image_stem,_image_bundle(job,row,recipe,camera,frame,original)),
                 (root/'lidar',lidar_stem,_lidar_bundle({**row,'label_path':label_path},camera,frame,original,record['points_path']))]
        result=dict(frame_id=frame,source_frame_id=original)
        for directory,stem,bundle in bundles:
            kind='lidar' if directory.name=='lidar' else 'image'
            if bundle is None:result[kind]='not_available';continue
            content,artifacts=bundle
            result[kind]=publish(directory,stem,content,{stem+suffix:value for suffix,value in artifacts.items()},backup)
        report['frames'].append(result)
    if root.is_dir():
        temporary=root/('.classes_'+uuid4().hex+'.tmp');temporary.write_bytes(classes_data);temporary.replace(class_path)
    if not root.is_dir():report['status']='not_available'
    report['versions']=str(backup) if backup.is_dir() else None
    report['protected_frames']=sum(any(str(v).startswith('protected_') for v in row.values()) for row in report['frames'])
    (job/'label_export.json').write_bytes(json_bytes(report))
    print('标签目录：'+str(root) if report['status']=='complete' else '尚无可导出的标签；仅保留原始预览。',flush=True)
    if report['protected_frames']:print(f"已保护 {report['protected_frames']} 帧已有标签，详见 label_export.json",flush=True)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--job',type=Path,required=True);parser.add_argument('--output',type=Path)
    args=parser.parse_args(argv);print(json.dumps(export_labels(args.job,args.output),ensure_ascii=False,indent=2))
