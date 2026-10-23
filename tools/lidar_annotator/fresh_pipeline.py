"""Raw-sequence annotation jobs; isolated inputs, inference, temporal fitting and review."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
from .cli import REPO, BACKEND, STAGES, execute, print_run_summary
from .lmdb_input import preview_frame
from .raw_input import frame_records, read_raw, input_format as detect_input_format
from .output_locations import visualization_location
from .stage_timing import measure_stage
from .frame_selection import parse_frames,debug_output
from .recipe_config import load_recipe,class_prompts


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp.json');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2));temp.replace(path)


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def _link_preview(source,destination):
    source=Path(source);destination=Path(destination)
    if not source.is_file():raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_name(f'.{destination.name}.{os.getpid()}.tmp')
    try:
        temporary.symlink_to(os.path.relpath(source,destination.parent))
        os.replace(temporary,destination)
    finally:
        if temporary.is_symlink():temporary.unlink()


def config(job):return json.loads((Path(job)/'backend.json').read_text())


def prepare(job):
    job=Path(job);raw=job/'raw';index=job/'data/labels/zgc/index.json';temporal=job/'temporal'
    c=dict(source_index=str(index),person_root=str(job/'data/person'),vehicle_root=str(job/'data/vehicle'),pose_npz=str(temporal/'poses.npz'),pose_report=str(temporal/'pose_report.json'),odometry_npz=str(raw/'odometry.npz'),temporal_output=str(temporal),vehicle_strategy='local_odom_anchor',pose_source='recorded_odom',actor_mask_source='sam3_video',coordinate_assumption='Recorded xyz/yaw, gravity-aligned LiDAR/body identity assumed; no IMU, roll/pitch or independent odom drift correction.',temporal_parameters=dict(local_anchor_max_age_s=2.))
    raw_report=raw/'report.json'
    if raw_report.is_file() and json.loads(raw_report.read_text()).get('input_format')=='kitti':
        c['coordinate_assumption']='KITTI camera-reference poses converted to the LiDAR reference frame using calibration; full supplied rigid transforms retained.'
    write(job/'backend.json',c);return c


def backend(job,name,*args):
    execute([sys.executable,str(BACKEND/name),'--config',str(Path(job)/'backend.json'),*map(str,args)],cwd=str(REPO))


def poses(job):
    c=config(job);sys.path.insert(0,str(BACKEND))
    from prepare_recorded_odom import pose_matrices
    with np.load(c['odometry_npz']) as z:
        p=z['poses'];times=z['timestamps'];supplied=z['transforms'].copy() if 'transforms' in z else None
    if np.shape(times)!=(len(p),) or not np.isfinite(times).all():raise ValueError('序列缺少有效的逐帧时间戳，无法时序优化；原始投影查看仍可使用')
    if supplied is not None:
        if supplied.shape!=(len(p),4,4) or not np.isfinite(supplied).all():raise ValueError('序列缺少有效的逐帧位姿，无法时序优化；原始投影查看仍可使用')
        matrices=np.linalg.inv(supplied[0])[None]@supplied
    else:matrices=pose_matrices(p)
    segments=np.zeros(len(p),np.int32);diagnostics=[]
    for i in range(len(p)):
        valid=True
        if i:
            dt=(int(times[i])-int(times[i-1]))*1e-9
            rotation=matrices[i-1,:3,:3].T@matrices[i,:3,:3]
            angle=np.arccos(np.clip((np.trace(rotation)-1)/2,-1,1))
            valid=0<dt<=.5 and np.linalg.norm(matrices[i,:3,3]-matrices[i-1,:3,3])<=1 and angle<=np.deg2rad(10)
            segments[i]=segments[i-1]+int(not valid)
        diagnostics.append(dict(frame=f'{i:06d}',valid=bool(valid),source='recorded odom'))
    root=Path(c['temporal_output']);root.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(c['pose_npz'],transforms=matrices,segments=segments)
    write(c['pose_report'],dict(method='supplied rigid transforms rebased to first frame' if supplied is not None else 'recorded xyz/yaw rebased to first frame',frames=len(p),segments=int(segments[-1])+1,diagnostics=diagnostics,coordinate_assumption=c['coordinate_assumption'],source_sha256=sha(c['odometry_npz'])))


def validate(job):
    c=config(job);source=Path(c['source_index']).parent;rows=json.loads(Path(c['source_index']).read_text())['rows'];counts=[]
    sys.path.insert(0,str(REPO/'tools/autolabel/semantic_zgc'))
    from centerpoint_weak_semantics import weak_targets
    for row in rows:
        path=source/row['path']
        if sha(path)!=row['sha256']:raise ValueError(f'Point label checksum mismatch: {path}')
        with np.load(path) as z:
            n=len(z['xyz']);labels=z['point_labels'];conf=z['point_confidence']
            assert labels.shape==conf.shape==(n,)
            assert np.isfinite(conf).all() and ((conf>=0)&(conf<=1)).all()
            assert (conf[labels==255]==0).all()
            targets=weak_targets(z,schema='zgc_image_teacher_seven_v2')
            for key,value in targets.items():np.testing.assert_array_equal(z[key],value)
            counts.append(n)
    backend(job,'audit_person_consistent_tracks.py')
    write(Path(job)/'validation.json',dict(status='passed',frames=len(rows),point_count=sum(counts),population='all job frames and points; no sampling',source_readonly=True,manual_ground_truth=False))


def render(job,workers):
    from PIL import Image,ImageDraw
    c=config(job);source=Path(c['source_index']).parent
    if not (source/'index.json').exists():
        source=Path(job)/'raw'
    rows=json.loads((source/'index.json').read_text())['rows'];out=Path(job)/'review';dest=out/'visualization';dest.mkdir(parents=True,exist_ok=True)
    optimized=Path(c['temporal_output'])/'map_review/sam3_full_video/interior_half_review/consistent_tracks/centers_only.npz'
    if optimized.is_file():
        local=Path(job)/'review_local';backend(job,'render_consistent_combined.py','--workers',workers,'--output',local)
        for row in rows:
            stem=Path(row['path']).stem;original=int(row.get('original_frame_id',row['frame_id']))
            for suffix in ('.png','.json'):
                _link_preview(local/'visualization'/f'{stem}{suffix}',dest/f'{original:06d}{suffix}')
    else:
        for row in rows:
            with np.load(source/row['path']) as z:a={k:z[k] for k in z.files}
            a['in_image']=a.get('visible',a['in_image']);image=Image.open(row['image_path']).convert('RGB');original=int(row.get('original_frame_id',row['frame_id']))
            preview_frame(image,a,dest/f'{original:06d}.png',original)
    html='<!doctype html><meta charset="utf-8"><title>LiDAR annotation review</title><style>body{background:#14171c;color:white}img{max-width:100%}</style><h1>逐帧标注审核</h1><p>自动标注候选；保留视觉 mask、原始观测与优化来源，待人工审核。</p>'
    ids=[int(r.get('original_frame_id',r['frame_id'])) for r in rows]
    html+=''.join(f'<h2>{f:06d}</h2><img loading="lazy" src="visualization/{f:06d}.png">' for f in ids)
    (out/'index.html').write_text(html);write(out/'report.json',dict(status='complete',frames=len(ids),frame_ids=ids,optimized=optimized.is_file()))


def catalog(job):
    c=config(job);source=Path(c['source_index']).parent
    if not (source/'index.json').exists():source=Path(job)/'raw'
    rows=json.loads((source/'index.json').read_text())['rows'];root=Path(c['temporal_output']);video=root/'map_review/sam3_full_video';person=video/'interior_half_review/consistent_tracks';out=Path(job)/'labels';out.mkdir(exist_ok=True)
    optimized=all(p.is_file() for p in (person/'centers_only.npz',root/'map_review/history_fit/bev_dedup/vehicle_map_boxes.npz',Path(c['pose_npz'])))
    if optimized:
        from tools.autolabel.zgc_labeling.finalize_person_positions import load_positions
        persons=load_positions(person/'centers_only.npz')
        with np.load(root/'map_review/history_fit/bev_dedup/vehicle_map_boxes.npz') as z:cars=z['box_3d_map'].reshape(-1,7);car_ids=z['track_ids']
        with np.load(c['pose_npz']) as z:transforms=z['transforms']
    records=[]
    for local,row in enumerate(rows):
        original=int(row.get('original_frame_id',row['frame_id']));stem=f'{local:06d}'
        with np.load(source/row['path']) as z:a={k:z[k] for k in z.files}
        if 'point_raw_labels' in a:
            from .display_range import DEFAULT_LIMITS
            from .track_preview import load_targets,person_support_audit
            local_stem=Path(row['path']).stem
            instance_path=Path(job)/'data/visual_instances'/f'{local_stem}.npz'
            records_path=instance_path.with_suffix('.json')
            visual_records=json.loads(records_path.read_text()).get('records',[]) if records_path.is_file() else []
            ids_path=video/'native_to_display_ids.json'
            display_ids=json.loads(ids_path.read_text()) if ids_path.is_file() else {}
            for record in visual_records:
                if record.get('track_id') is not None:
                    record['display_id']=display_ids.get(str(record['track_id']),record['track_id'])
            paths=dict(tracks=person/'centers_only.npz',local=local,
                       mask=source/'image_masks'/f'{local_stem}.png',
                       person_metadata=Path(c['person_root'])/'metadata'/f'{local_stem}.json',id_map=ids_path,
                       instance_masks=instance_path,track_masks=video/'masks'/f'{local_stem}.npz')
            targets=load_targets(paths,a,visual_records,DEFAULT_LIMITS)
            audit=person_support_audit(targets)
            ignored=np.asarray(audit['ignored_indices'],dtype=np.int64)
            ignored=ignored[(ignored>=0)&(ignored<len(a['point_raw_labels']))]
            ignored=ignored[a['point_raw_labels'][ignored]==1]
            for instance in audit['instances']:
                indices=np.asarray(instance['ignored_indices'],dtype=np.int64)
                indices=indices[(indices>=0)&(indices<len(a['point_raw_labels']))]
                instance['ignored_indices']=indices[a['point_raw_labels'][indices]==1].tolist()
            audit['ignored_indices']=np.unique(ignored).tolist()
            a['point_raw_labels']=np.asarray(a['point_raw_labels']).copy()
            a['point_raw_labels'][ignored]=255
            if 'point_raw_confidence' in a:
                a['point_raw_confidence']=np.asarray(a['point_raw_confidence']).copy();a['point_raw_confidence'][ignored]=0
            if 'point_labels' in a:
                human=ignored[a['point_labels'][ignored]==1]
                a['point_labels']=np.asarray(a['point_labels']).copy();a['point_labels'][human]=255
                if 'point_confidence' in a:
                    a['point_confidence']=np.asarray(a['point_confidence']).copy();a['point_confidence'][human]=0
            a['person_semantic_ignored_indices']=ignored.astype(np.int64)
            a['person_semantic_filter_json']=np.asarray(json.dumps(audit,ensure_ascii=False,separators=(',',':')))
            if len(ignored) and str(np.asarray(a.get('supervision_policy','')).item())=='static_drivable_occupancy_v1' and 'weak_occupancy_bits' in a:
                from .fresh_semantics import weak_targets,teacher_bev_from_occupancy_bits
                target=weak_targets(a,schema='zgc_image_teacher_seven_v2');a.update(target)
                bits=target['weak_occupancy_bits'];occupancy=np.full((320,320),255,np.uint8)
                for label,occ in enumerate((0,1,2,5,4,6,3)):
                    occupancy[bits==(1<<occ)]=occ
                a['occupancy_labels']=occupancy;a['bev_labels']=teacher_bev_from_occupancy_bits(bits)
        if optimized:
            T=transforms[local];inverse=np.linalg.inv(T);p=(persons['frame_ids']==local)&persons['position_available'];boxes=cars.copy()
            if len(boxes):boxes[:,:3]=boxes[:,:3]@inverse[:3,:3].T+inverse[:3,3];boxes[:,6]-=np.arctan2(T[1,0],T[0,0])
            keep=(np.abs(boxes[:,:2])<20).all(1) if len(boxes) else np.zeros(0,bool)
            a.update(person_xyz=persons['corrected_point_lidar'][p],person_track_ids=persons['sam_video_ids'][p],person_position_source=persons['position_source'][p],vehicle_box_3d=boxes[keep],vehicle_track_ids=car_ids[keep])
            a.update(person_track_only=persons['track_only'][p],person_position_policy=np.asarray(persons['position_policy']))
            if 'selected_support_counts' in persons:a['person_support_counts']=persons['selected_support_counts'][p]
        a['manual_ground_truth']=np.asarray(False)
        path=out/f'{original:06d}.npz';np.savez_compressed(path,**a)
        records.append(dict(frame_id=original,local_frame=local,label=str(path.relative_to(job)),label_sha256=sha(path),image=row['image_path'],semantic_mask=str(source/'image_masks'/f'{stem}.png'),semantic_confidence=str(source/'confidence'/f'{stem}.npz'),visual_instances=str(Path(job)/'data/visual_instances'/f'{stem}.npz'),person_metadata=str(Path(c['person_root'])/'metadata'/f'{stem}.json'),vehicle_metadata=str(Path(c['vehicle_root'])/'refined'/f'{stem}.json'),video_masks=str(video/'masks'/f'{stem}.npz'),visual_track_associations=str(video/'associations'/f'{stem}.json'),review_image=str(Path(job)/'review/visualization'/f'{original:06d}.png')))
        records[-1]['missing_artifacts']=[str(value) for key,value in records[-1].items() if key in ('semantic_mask','semantic_confidence','visual_instances','person_metadata','vehicle_metadata','video_masks','visual_track_associations','review_image') and not Path(value).is_file()]
    recipe=load_recipe(Path(job)/'recipe.json')
    result=dict(schema='lidar_fresh_annotations_v1',optimized_positions_available=optimized,recipe='recipe.json',classes=recipe['raw_classes'],disabled_classes=recipe.get('disabled_classes',[]),frames=len(records),manual_ground_truth=False,labels_promoted=False,source='job.json',coordinate_assumption=c['coordinate_assumption'],rows=records)
    write(Path(job)/'annotations.json',result)
    from .dataset_labels import export_labels
    export_labels(job)



def prepare_tracked_observations(job):
    """Export already tracked actor masks; never invoke a visual model here."""
    from .actor_video import read_prompt,prompt_source_signature
    c=config(job);job=Path(job);recipe=load_recipe(job/'recipe.json');source=Path(c['source_index']).parent;rows=json.loads((source/'index.json').read_text())['rows']
    video=Path(c['temporal_output'])/'map_review/sam3_full_video';(video/'masks').mkdir(parents=True,exist_ok=True);(video/'associations').mkdir(exist_ok=True)
    prompts=[] if 1 in recipe.get('disabled_classes',[]) else recipe['base_semantics']['prompts']['1']
    signatures={p:prompt_source_signature(job,p,recipe) for p in prompts};matched=total=0
    if any(v is None for v in signatures.values()):raise ValueError('SAM3 actor cache missing; execute semantic stage first')
    id_map_path=video/'native_to_display_ids.json'
    id_map=json.loads(id_map_path.read_text()) if id_map_path.exists() else {}
    for i,row in enumerate(rows):
        stem=Path(row['path']).stem;parts=[read_prompt(job,p,i,recipe) for p in prompts]
        if parts:
            masks=np.concatenate([x['masks'] for x in parts]);ids=np.array([r['track_id'] for x in parts for r in x['records']],np.int64);shape=parts[0]['evidence'].shape
        else:
            from PIL import Image
            image=Image.open(row['image_path']);shape=(image.height,image.width);masks=np.empty((0,*shape),bool);ids=np.empty(0,np.int64)
        native_ids=ids.copy()
        for tid in native_ids:
            if str(int(tid)) not in id_map:id_map[str(int(tid))]=len(id_map)
        ids=np.array([id_map[str(int(tid))] for tid in native_ids],np.int64)
        np.savez_compressed(video/'masks'/f'{stem}.npz',object_ids=ids,native_object_ids=native_ids,mask_shape=shape,masks_packed=np.packbits(masks.reshape(len(ids),int(np.prod(shape))),axis=1,bitorder='little'))
        person=Path(c['person_root']);candidates=json.loads((person/'raw_sam_person/metadata'/f'{stem}.json').read_text())['candidates']
        with np.load(person/'frames'/f'{stem}.npz') as z:rawids=z['xy_instance_ids'];xy=z['xy']
        observations=[]
        for iid,point in zip(rawids,xy):
            tid=candidates[int(iid)].get('track_id');valid=tid is not None and int(tid) in native_ids
            observations.append(dict(instance_id=int(iid),raw_xy=point.tolist(),sam_video_id=id_map[str(int(tid))] if valid else None,native_sam3_track_id=int(tid) if valid else None,mask_iou=1.0 if valid else None))
            matched+=int(valid);total+=1
        write(video/'associations'/f'{stem}.json',dict(frame=int(stem),observations=observations))
    write(id_map_path,id_map)
    print('tracking cache: reused SAM3 video masks and IDs; visual model calls=0',flush=True)
    write(video/'report.json',dict(status='complete',frame_count=len(rows),raw_positions=total,matched_raw_positions=matched,identity_source='cached native SAM3 actor-video IDs; no model inference in tracking stage',source_signatures=signatures))
    audit=video/'object_category_audit.json'
    if audit.exists():
        previous=json.loads(audit.read_text())
        if previous.get('source_signatures')!=signatures:
            write(video/'previous_object_category_audit.json',previous);write(audit,dict(excluded_objects=[],review_status='not_manually_reviewed',source_signatures=signatures))
    else:write(audit,dict(excluded_objects=[],review_status='not_manually_reviewed',source_signatures=signatures))

def _stage(job,name,workers,refresh=False):
    os.environ['LIDAR_JOB_OUTPUT']=str(Path(job).resolve())
    from tools.autolabel.task_progress import report_progress
    report_progress(name,0,0,detail='initializing',job=job)
    if name=='semantic':
        from .actor_video import run as track_actors
        recipe=load_recipe(Path(job)/'recipe.json')
        disabled=set(recipe.get('disabled_classes',[]))
        for item in recipe['raw_classes']:
            cid=item['id'];status='停用' if cid in disabled else '启用'
            print(f"类别 {cid} · {item['name']} · {status} · 提示词："+' / '.join(class_prompts(recipe,cid)),flush=True)
        settings=json.loads((Path(job)/'semantic_options.json').read_text()) if (Path(job)/'semantic_options.json').exists() else {}
        prompts=None
        if settings.get('classes') is not None:
            prompts=[]
            for cid in settings['classes']:
                prompts.extend(class_prompts(recipe,cid))
                if cid==0:
                    prompts.extend(recipe['base_semantics']['grass_prompts']+recipe['ground_refinement']['prompts'])
                    prompts.extend(recipe[k]['prompt'] for k in ('manhole_covers','small_debris'))
        actor_report=track_actors(job,recipe,refresh=refresh,refresh_prompts=prompts)
        if isinstance(actor_report,dict):
            for prompt,info in actor_report.get('prompts',{}).items():print(f"SAM3视频 {prompt!r}：复用 {info['reused_chunks']}/{info['chunks']} 个片段",flush=True)
        from .fresh_semantics import run
        run(job,refresh=refresh,refresh_prompts=prompts)
    elif name=='instances':
        from tools.autolabel.fresh_instances import run
        run(job,refresh=refresh)
    elif name=='temporal':poses(job);backend(job,'temporal_refine.py')
    elif name=='tracking':
        prepare_tracked_observations(job);backend(job,'prepare_person_observations.py')
    elif name=='optimize':
        for file in ('visualize_vehicle_map.py','fit_vehicle_map.py','deduplicate_vehicle_map.py','refine_vehicle_map.py','run_person_consistent_tracks.py','finalize_person_positions.py'):backend(job,file)
    elif name=='validate':validate(job)
    elif name=='render':render(job,workers)
    elif name=='catalog':catalog(job)



def stage(job,name,workers,refresh=False):
    from .stage_cache import reuse,complete
    job=Path(job).resolve()
    cached=name!='semantic' and reuse(job,name,force=refresh)
    if cached:
        print(f'{name}: cached outputs verified; no repeated computation',flush=True)
        if name=='catalog':
            from .dataset_labels import export_labels
            export_labels(job)
    else:_stage(job,name,workers,refresh)
    if name=='render':
        from .optimized_map import optimized_map_ready,load_optimized_map
        context=dict(output=str(job),backend=config(job))
        if optimized_map_ready(context):
            def progress(done,total):
                from tools.autolabel.task_progress import report_progress
                report_progress('render',done,total,detail='生成全帧地图',job=job)
            data=load_optimized_map(context,progress=progress)
            print('全帧地图：'+('复用已保存结果' if data['cache_reused'] else '已生成并保存')+' · '+data['directory'],flush=True)
        else:print('全帧地图：缺少完整优化结果，本次跳过',flush=True)
    if name!='semantic' and not cached:complete(job,name)

def run(source,output,stages=STAGES,workers=6,frames=None,dry_run=False,refresh_semantic=False,refresh_instances=False,recipe_path=None,semantic_classes=None,input_format=None,camera=None,dataset_output=None,refresh_postprocess=False):
    if refresh_postprocess:
        if refresh_semantic or refresh_instances:raise ValueError('复用 SAM 重算后处理不能同时强制重新推理')
        stages=[s for s in stages if s not in ('semantic','instances')]
        if not stages:raise ValueError('请选择至少一个后处理阶段')
    started_unix=time.time()
    detected=detect_input_format(source)
    if camera is not None and detected!='kitti':raise ValueError('--camera 仅用于 KITTI 相机通道')
    if input_format is not None and input_format!=detected:raise ValueError('所选数据格式与输入目录不符')
    source,records=frame_records(source,camera=camera);frames=parse_frames(frames);job=debug_output(output,frames)
    if job==source or source in job.parents:raise ValueError('Output must be outside source recording')
    if job==REPO or REPO in job.parents:raise ValueError('工作目录必须位于源码目录之外')
    selected=set(frames) if frames is not None else None
    recipe=load_recipe(recipe_path if recipe_path else job/'recipe.json' if (job/'recipe.json').exists() else None)
    if refresh_postprocess:
        required=[job/'recipe.json',job/'data/labels/zgc/index.json',job/'data/person/index.json',job/'data/vehicle/index.json']
        if not all(path.is_file() for path in required):raise ValueError('缺少已有语义或实例结果，请先使用“复用缓存继续执行”完成完整流程')
        if recipe!=load_recipe(job/'recipe.json'):raise ValueError('提示词或标注配置已改变，请先使用“复用缓存继续执行”更新模型结果')
    if semantic_classes is not None and any(cid not in range(10) for cid in semantic_classes):raise ValueError('Semantic class IDs must be 0 through 9')
    ids=[int(r['global_frame_id']) for r in records if selected is None or int(r['global_frame_id']) in selected]
    if not ids or selected is not None and selected-set(ids):raise ValueError('Requested frames missing from recording')
    stages=[s for s in STAGES if s in stages]
    if not stages or workers<1:raise ValueError('Select stages and positive workers')
    identity=dict(schema='lidar_fresh_job_v1',source=str(source),frame_ids=ids,source_readonly=True)
    if detected=='kitti':identity.update(input_format='kitti',camera=records[0]['camera'])
    destination=Path(dataset_output).expanduser().resolve() if dataset_output is not None else job/'kitti'
    if dataset_output is not None:
        if detected!='lmdb':raise ValueError('独立数据导出目录仅用于 LMDB 输入')
        if destination==REPO or REPO in destination.parents:raise ValueError('标准数据目录必须位于源码目录之外')
        if job==destination or destination in job.parents or job in destination.parents:raise ValueError('数据目录和工作目录必须分开')
        if destination==source or source in destination.parents:raise ValueError('数据导出目录必须在原始 LMDB 之外')
        identity['dataset_output']=str(destination)
    owner=job/'job.json'
    if owner.exists() and json.loads(owner.read_text())!=identity:raise ValueError('Output belongs to a different source or frame selection; use a new job directory')
    plan=[dict(stage=s,commands=[[sys.executable,str(REPO/'lidar_annotator.py'),'annotate-stage','--job',str(job),'--stage',s,'--workers',str(workers)]+(['--refresh'] if s=='semantic' and refresh_semantic or s=='instances' and refresh_instances or refresh_postprocess else [])]) for s in stages]
    if dry_run:return dict(status='dry_run',job=identity,output=str(job),plan=plan)
    job.mkdir(parents=True,exist_ok=True)
    lock=(job/'.run.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close();raise ValueError('This job is already running') from None
    try:
        write(owner,identity)
        visualization=job/'review/visualization';visualization.mkdir(parents=True,exist_ok=True)
        print(f'可视化目录：{visualization}',flush=True)
        if dataset_output is not None:print('标准数据目录：'+str(destination),flush=True)
        raw=job/'raw'
        if not (raw/'report.json').exists():
            imported=read_raw(source,raw,frames=ids,camera=camera,kitti_output=destination)
            if detected=='lmdb' and imported.get('kitti_export_status')!='complete':raise ValueError('标准数据导出失败：'+imported.get('kitti_export_error','请查看导出日志'))
        elif detected=='lmdb':
            from .kitti_input import export_imported_kitti
            def conversion_progress(done,total):
                write(raw/'status.json',dict(status='running',phase='kitti',completed=done,total=total))
                print(f'KITTI 转换：{done}/{total} 帧',flush=True)
            try:
                export_imported_kitti(raw,destination,progress=conversion_progress)
            except KeyboardInterrupt:
                write(raw/'status.json',dict(status='cancelled',phase='kitti',error='KITTI conversion cancelled'));raise
            except (OSError,ValueError,KeyError) as exc:
                write(raw/'status.json',dict(status='failed',phase='kitti',error=str(exc),kitti_export_status='failed',kitti_export_error=str(exc)));raise
            imported=json.loads((raw/'report.json').read_text());imported.update(kitti_export_status='complete',kitti_output=str(destination));imported.pop('kitti_export_error',None)
            write(raw/'report.json',imported);write(raw/'status.json',imported)
        if detected=='lmdb' and not (destination/'metadata.npz').is_file():raise ValueError('标准数据导出未完成，请查看 raw/report.json 中的导出错误')
        from .camera_coverage import generate_coverage
        generate_coverage(job)
        write(job/'recipe.json',recipe);write(job/'semantic_options.json',dict(classes=semantic_classes))
        prepare(job);(job/'logs').mkdir(exist_ok=True);report=dict(status='running',stages=[],frame_ids=ids,started_unix=started_unix,labels_promoted=False)
        entry=None
    except BaseException:
        lock.close();raise
    try:
        for item in plan:
            entry=dict(stage=item['stage'],status='running',commands=item['commands']);report['stages'].append(entry);print(f"Starting {item['stage']}",flush=True)
            with measure_stage(entry):
                write(job/'run_report.json',report)
                location,_=visualization_location(dict(output=str(job),review_output=str(job/'review'),backend=config(job)),item['stage'])
                if location is not None:print(f'可视化目录：{location}',flush=True)
                with (job/'logs'/f"{item['stage']}.log").open('a') as log:
                    for command in item['commands']:execute(command,cwd=str(REPO),stdout=log,stderr=subprocess.STDOUT)
            entry['status']='complete';write(job/'run_report.json',report)
            print(f"阶段耗时 · {item['stage']}：{entry['elapsed_seconds']:.3f} 秒",flush=True)
        report['status']='complete'
    except BaseException as exc:
        report['status']='cancelled' if isinstance(exc,KeyboardInterrupt) else 'failed';report['error']=str(exc)
        if entry:entry['status']=report['status']
        raise
    finally:
        report['elapsed_seconds']=round(time.time()-report['started_unix'],3);write(job/'run_report.json',report)
        lock.close()
    return report


def main(argv=None,worker=False):
    p=argparse.ArgumentParser(description=__doc__)
    if worker:
        p.add_argument('--job',type=Path,required=True);p.add_argument('--stage',choices=STAGES,required=True);p.add_argument('--workers',type=int,default=6);p.add_argument('--refresh',action='store_true');a=p.parse_args(argv)
        def cancel_worker(signum,frame):raise KeyboardInterrupt('Annotation stage cancelled')
        signal.signal(signal.SIGTERM,cancel_worker)
        stage(a.job,a.stage,a.workers,a.refresh);return
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--stages',choices=STAGES,nargs='+',default=list(STAGES));p.add_argument('--workers',type=int,default=6);p.add_argument('--frames',nargs='+');p.add_argument('--recipe',type=Path);p.add_argument('--semantic-classes',nargs='+',type=int);p.add_argument('--dry-run',action='store_true');p.add_argument('--refresh-semantic',action='store_true');p.add_argument('--refresh-instances',action='store_true');p.add_argument('--refresh-postprocess',action='store_true',help='复用已有模型结果，强制重算所选后处理阶段');p.add_argument('--quiet-report',action='store_true');p.add_argument('--input-format',choices=['lmdb','kitti']);p.add_argument('--work-dir',type=Path,help='独立工作缓存目录，LMDB 的 --output 为标准数据目录');p.add_argument('--camera',help='KITTI image_* channel');a=p.parse_args(argv)
    def cancel(signum,frame):raise KeyboardInterrupt('Annotation cancelled')
    old=signal.signal(signal.SIGTERM,cancel)
    try:
        output=a.output;destination=None
        if detect_input_format(a.source)=='lmdb':
            from .job_paths import work_directory
            destination=a.output;output=a.work_dir or work_directory(a.source,a.output,parse_frames(a.frames))
        elif a.work_dir is not None:raise ValueError('--work-dir 仅用于 LMDB 输入')
        result=run(a.source,output,a.stages,a.workers,a.frames,a.dry_run,a.refresh_semantic,a.refresh_instances,a.recipe,a.semantic_classes,a.input_format,a.camera,destination,a.refresh_postprocess)
        if a.quiet_report and not a.dry_run:print_run_summary(result,debug_output(output,parse_frames(a.frames)))
        else:print(json.dumps(result,ensure_ascii=False,indent=2))
    finally:signal.signal(signal.SIGTERM,old)
