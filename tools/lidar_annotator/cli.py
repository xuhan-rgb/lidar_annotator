"""Unified semantic, visual instance, LiDAR instance, tracking and review stages."""
import argparse
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
BACKEND=REPO/'tools/autolabel/zgc_labeling'
STAGES=('semantic','instances','temporal','tracking','optimize','validate','render','catalog')
DEFAULT=REPO/'tools/autolabel/zgc_mask_recipe_v1/recipe.json'


def load_config(path):
    path=Path(path).resolve();c=json.loads(path.read_text())
    if c['adapter']!='zgc_cached_v1':raise ValueError('Only the existing ZGC cache adapter is implemented')
    for key in ('backend_config','python','sam_python','sam_source','semantic_runner','output'):
        value=Path(c[key]).expanduser();c[key]=os.path.abspath(path.parent/value) if not value.is_absolute() else str(value)
    if os.environ.get('LIDAR_RUNTIME_PYTHON'):c['python']=c['sam_python']=os.environ['LIDAR_RUNTIME_PYTHON']
    c['profile']=str(path);c['backend']=json.loads(Path(c['backend_config']).read_text())
    review=Path(c.get('review_output',Path(c['output'])/'review'));c['review_output']=str((path.parent/review).resolve() if not review.is_absolute() else review)
    c['visual_sources']={k:str((path.parent/Path(v)).resolve() if not Path(v).is_absolute() else Path(v)) for k,v in c.get('visual_sources',{}).items()}
    return c


def commands(c,stage,workers=6,frames=None,refresh_semantic=False,refresh_instances=False):
    py=c['python'];cfg=['--config',c['backend_config']]
    def backend(name,*args):return [py,str(BACKEND/name),*map(str,args)]
    def own(*args):return [py,str(HERE/'cli.py'),*args,'--config',c['profile']]
    if stage=='semantic':
        return ([[c['sam_python'],c['semantic_runner']]] if refresh_semantic else [])+[own('check','--scope','semantic')]
    if stage=='instances':
        generated=[]
        if refresh_instances:
            generated=[[py,str(REPO/'tools/autolabel/pedestrian_instance_zgc/process_instances.py'),'--all'],[py,str(REPO/'tools/autolabel/wilddet3d_zgc/generate_vehicle_camera_ray_distance030_batch.py'),'--workers',str(workers)]]
        return generated+[own('check','--scope','instances')]
    if stage=='temporal':return [backend('prepare_recorded_odom.py',*cfg),backend('temporal_refine.py',*cfg)]
    if stage=='tracking':
        video=Path(c['backend']['temporal_output'])/'map_review/sam3_full_video'
        cached=(video/'report.json').is_file() and json.loads((video/'report.json').read_text()).get('status')=='complete'
        if not cached and (video/'object_category_audit.json').is_file() and json.loads((video/'object_category_audit.json').read_text()).get('excluded_objects'):
            raise ValueError('Existing manual category exclusions belong to old SAM IDs; use a new temporal_output for a fresh tracking run')
        tracking=[own('check','--scope','tracking')] if cached else [[c['sam_python'],str(BACKEND/'run_sam3_full_video.py'),*cfg]]
        return tracking+[backend('prepare_person_observations.py',*cfg)]
    if stage=='optimize':return [backend(name,*cfg) for name in ('visualize_vehicle_map.py','fit_vehicle_map.py','deduplicate_vehicle_map.py','refine_vehicle_map.py','run_person_consistent_tracks.py','finalize_person_positions.py')]
    if stage=='validate':return [backend('audit_person_consistent_tracks.py',*cfg),own('check','--scope','optimized')]
    if stage=='render':
        args=['--workers',str(workers),'--output',c['review_output']]
        if frames is not None:args+=['--frames',*map(str,frames)]
        return [backend('render_consistent_combined.py',*cfg,*args)]
    if stage=='catalog':return [own('catalog')]
    raise ValueError(stage)


def check(c,scope):
    b=c['backend'];source=Path(b['source_index']);root=Path(b['temporal_output']);rows=json.loads(source.read_text())['rows'];frames=[Path(r['path']).stem for r in rows]
    if scope in ('semantic','instances'):
        sys.path.insert(0,str(BACKEND))
        from pipeline import _validate_semantic_cache,_validate_person_cache,_validate_vehicle_cache
        if scope=='semantic':return _validate_semantic_cache(source)
        return dict(person=_validate_person_cache(source,b['person_root']),vehicle=_validate_vehicle_cache(source,b['vehicle_root']))
    if scope in ('tracking','optimized'):
        person=root/'map_review/sam3_full_video/interior_half_review/consistent_tracks'
        for f in frames:
            for path in [source.parent/'frames'/f'{f}.npz',Path(b['person_root'])/'raw_sam_person/frames'/f'{f}.npz',root/'map_review/sam3_full_video/masks'/f'{f}.npz',root/'map_review/sam3_full_video/associations'/f'{f}.json']:
                if not path.is_file():raise FileNotFoundError(path)
        if scope=='tracking':
            report=json.loads((root/'map_review/sam3_full_video/report.json').read_text())
            if report['status']!='complete' or report['frame_count']!=len(frames):raise ValueError('Incomplete SAM3 video report')
            return dict(status='passed',frames=len(frames),visual_tracking_cache_retained=True)
        audit=json.loads((person/'integrity_audit.json').read_text())
        if audit['independent_local_linear_outliers'] or audit['adjacent_jump_outliers']:raise ValueError('Optimized positions failed continuity audit')
        for file in ['report.json','centers_only.npz']:
            if not (person/file).is_file():raise FileNotFoundError(person/file)
        if not (root/'map_review/history_fit/bev_dedup/vehicle_map_boxes.npz').is_file():raise FileNotFoundError('Deduplicated vehicle map missing')
        return dict(status='passed',frames=len(frames),retained_person_positions=audit['retained'],visual_masks_and_associations_preserved=True)
    raise ValueError(scope)


def catalog(c):
    """Index original visual annotations and metric annotations without lossy conversion."""
    b=c['backend'];root=Path(b['temporal_output']);source=Path(b['source_index']);rows=json.loads(source.read_text())['rows'];out=Path(c['output']);out.mkdir(parents=True,exist_ok=True)
    person=Path(b['person_root']);vehicle=Path(b['vehicle_root']);video=root/'map_review/sam3_full_video';optimized=video/'interior_half_review/consistent_tracks';items=[]
    for row in rows:
        stem=Path(row['path']).stem
        visual=dict(semantic_mask=source.parent/'image_masks'/f'{stem}.png',semantic_confidence=source.parent/'confidence'/f'{stem}.npz',semantic_metadata=source.parent/'metadata'/f'{stem}.json',person_instance_masks=person/'raw_sam_person/frames'/f'{stem}.npz',person_detection_metadata=person/'metadata'/f'{stem}.json',sam_video_masks=video/'masks'/f'{stem}.npz',visual_track_associations=video/'associations'/f'{stem}.json',vehicle_detection_metadata=vehicle/'refined'/f'{stem}.json')
        visual.update({name:Path(folder)/f'{stem}.json' for name,folder in c.get('visual_sources',{}).items()})
        if 'vehicle_masks' in c.get('visual_sources',{}):visual['vehicle_instance_masks']=Path(c['visual_sources']['vehicle_masks'])/f'{stem}.npz'
        metric=dict(source_point_semantics_and_calibration=source.parent/row['path'],original_person_positions_and_supports=person/'frames'/f'{stem}.npz',original_vehicle_boxes=vehicle/'refined'/f'{stem}.npz',vehicle_support_indices=vehicle/'refined/support'/f'{stem}.npz')
        record=dict(frame=stem,image_source=row,visual={k:str(v) for k,v in visual.items()},lidar={k:str(v) for k,v in metric.items()},pedestrian_optimization_metadata=[str(p) for p in sorted((optimized/'metadata').glob(f'{stem}_I*.json'))],review_image=str(Path(c['review_output'])/'visualization'/f'{stem}.png'))
        record['missing_artifacts']=[str(p) for p in [*visual.values(),*metric.values()] if not p.is_file()];items.append(record)
    result=dict(schema='lidar_annotation_catalog_v1',adapter=c['adapter'],frames=len(items),manual_ground_truth=False,added_to_training=False,storage='Reference catalog; original files are retained in place. Copy referenced files with this catalog when migrating.',recipe=str(REPO/'tools/autolabel/zgc_mask_recipe_v1/recipe.json'),poses=b['pose_npz'],optimized_person_centres=str(optimized/'centers_only.npz'),deduplicated_vehicle_map=str(root/'map_review/history_fit/bev_dedup/vehicle_map_boxes.npz'),category_review=str(video/'object_category_audit.json'),rows=items)
    path=out/'annotations.json';path.write_text(json.dumps(result,ensure_ascii=False,indent=2));return dict(catalog=str(path),frames=len(items),frames_with_missing_artifacts=sum(bool(r['missing_artifacts']) for r in items))


def execute(command,**kwargs):
    """Cancel the whole worker group when the terminal or Qt controller stops a job."""
    kwargs.pop('check',None)
    process=subprocess.Popen(command,start_new_session=True,**kwargs)
    def terminate(signum,frame):
        raise KeyboardInterrupt('Annotation cancelled')
    previous=signal.signal(signal.SIGTERM,terminate)
    try:
        code=process.wait()
        if code:raise subprocess.CalledProcessError(code,command)
    except BaseException:
        if process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
        raise
    finally:signal.signal(signal.SIGTERM,previous)


def run(c,stages,workers=6,frames=None,refresh_semantic=False,refresh_instances=False,dry_run=False,runner=execute):
    if c.get('diagnostic_only') and any(s not in ('render','catalog') for s in stages):raise ValueError('Diagnostic frame bundles support render/catalog only; temporal optimization requires the full sequence')
    if frames is not None and tuple(stages)!=('render',):raise ValueError('--frames is supported only with --stages render; tracking and optimization use full sequences')
    if workers<1:raise ValueError('--workers must be positive')
    from .stage_timing import measure_stage
    plan=[dict(stage=s,commands=commands(c,s,workers,frames,refresh_semantic,refresh_instances)) for s in stages]
    if dry_run:return dict(status='dry_run',plan=plan)
    out=Path(c['output']);(out/'logs').mkdir(parents=True,exist_ok=True)
    report=dict(status='running',profile=c['profile'],stages=[],started_unix=time.time(),labels_promoted=False)
    env={**os.environ,'PYTHONPATH':os.pathsep.join([c['sam_source'],str(REPO/'tools/autolabel')]),'LIDAR_ANNOTATOR_PATHS':str(REPO/'configs/paths.source.json'),'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
    status=out/'run_report.json'
    try:
        for stage in plan:
            entry=dict(stage=stage['stage'],status='running',commands=stage['commands']);report['stages'].append(entry)
            with measure_stage(entry):
                status.write_text(json.dumps(report,indent=2))
                print(f"Starting {stage['stage']}",flush=True)
                with (out/'logs'/f"{stage['stage']}.log").open('a') as log:
                    for command in stage['commands']:runner(command,cwd=str(REPO),env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            entry['status']='complete'
            status.write_text(json.dumps(report,indent=2))
            print(f"阶段耗时 · {stage['stage']}：{entry['elapsed_seconds']:.3f} 秒",flush=True)
        report['status']='complete'
    except BaseException as exc:
        state='cancelled' if isinstance(exc,KeyboardInterrupt) else 'failed'
        entry['status']=state;report['status']=state;report['error']=str(exc);raise
    finally:
        report['elapsed_seconds']=round(time.time()-report['started_unix'],3);status.write_text(json.dumps(report,indent=2))
    return report


def print_run_summary(report,output):
    print(f"处理完成：{len(report['stages'])} 个步骤，用时 {report['elapsed_seconds']} 秒。")
    print(f"完整报告：{Path(output)/'run_report.json'}")


def main(argv=None):
    os.chdir(REPO)
    parser=argparse.ArgumentParser(prog='lidar-annotator',description=__doc__,epilog='Qt desktop: ./run.sh gui | Offline demo: ./run.sh demo | Runtime/GPU: ./run.sh doctor | Tests: ./run.sh self-test');sub=parser.add_subparsers(dest='command',required=True)
    for name in ('run','check','catalog'):
        p=sub.add_parser(name);p.add_argument('--config',type=Path,required=True)
        if name=='run':
            p.add_argument('--quiet-report',action='store_true');p.add_argument('--stages',choices=STAGES,nargs='+',default=list(STAGES));p.add_argument('--workers',type=int,default=6);p.add_argument('--frames',type=int,nargs='+');p.add_argument('--dry-run',action='store_true');p.add_argument('--refresh-semantic',action='store_true');p.add_argument('--refresh-instances',action='store_true')
        if name=='check':p.add_argument('--scope',choices=['semantic','instances','tracking','optimized'],required=True)
    args=parser.parse_args(argv);c=load_config(args.config)
    if args.command=='run':result=run(c,args.stages,args.workers,args.frames,args.refresh_semantic,args.refresh_instances,args.dry_run)
    elif args.command=='check':result=check(c,args.scope)
    else:result=catalog(c)
    if args.command=='run' and args.quiet_report and not args.dry_run:print_run_summary(result,c['output'])
    else:print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
