"""Build a movable diagnostic-frame bundle with images, visual labels and LiDAR."""
import argparse,hashlib,json,shutil,sys
from pathlib import Path
from .cli import REPO,load_config
from .storage import directory,sample_path
from tools.autolabel.resource_paths import dependency_root


def export(profile,frames,destination):
    import numpy as np
    c=load_config(profile);b=c['backend'];source=Path(b['source_index']).parent;root=Path(b['temporal_output']);person=Path(b['person_root']);vehicle=Path(b['vehicle_root']);dest=Path(destination).resolve()
    dest.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(REPO/'tools/autolabel/semantic_zgc'));from common import read_source_image
    rows=json.loads(Path(b['source_index']).read_text())['rows'];selected=[r for r in rows if int(Path(r['path']).stem) in frames]
    if len(selected)!=len(set(frames)):raise ValueError('Unknown or duplicate frame IDs')
    def copy(src,relative):
        dst=dest/relative;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    index=[]
    for row in selected:
        stem=Path(row['path']).stem;image=dest/'images'/f'{stem}.png';image.parent.mkdir(exist_ok=True);read_source_image(row).save(image)
        updated=dict(row,image_path=str(image),path=f'frames/{stem}.npz');index.append(updated)
        for folder,suffix in [('frames','.npz'),('image_masks','.png'),('confidence','.npz'),('metadata','.json')]:copy(source/folder/f'{stem}{suffix}',Path('source')/folder/f'{stem}{suffix}')
        for folder,suffix in [('frames','.npz'),('metadata','.json'),('raw_sam_person/frames','.npz'),('raw_sam_person/metadata','.json')]:copy(person/folder/f'{stem}{suffix}',Path('persons')/folder/f'{stem}{suffix}')
        for relative in [f'refined/{stem}.npz',f'refined/{stem}.json',f'refined/support/{stem}.npz']:copy(vehicle/relative,Path('vehicles')/relative)
        for folder,suffix in [('masks','.npz'),('associations','.json')]:copy(root/'map_review/sam3_full_video'/folder/f'{stem}{suffix}',Path('temporal/map_review/sam3_full_video')/folder/f'{stem}{suffix}')
        for file in (root/'map_review/sam3_full_video/interior_half_review/consistent_tracks/metadata').glob(f'{stem}_I*.json'):copy(file,Path('temporal/map_review/sam3_full_video/interior_half_review/consistent_tracks/metadata')/file.name)
        for name,folder in c['visual_sources'].items():
            copy(Path(folder)/f'{stem}.json',Path('visual')/name/f'{stem}.json')
            if name=='vehicle_masks':copy(Path(folder)/f'{stem}.npz',Path('visual')/name/f'{stem}.npz')
    (dest/'source/index.json').write_text(json.dumps(dict(rows=index),indent=2))
    copy(Path(b['pose_npz']),'temporal/poses.npz')
    for file in ['map_review/history_fit/report.json','map_review/history_fit/bev_dedup/report.json','map_review/history_fit/bev_dedup/vehicle_map_boxes.npz','map_review/sam3_full_video/object_category_audit.json']:copy(root/file,Path('temporal')/file)
    relative=Path('temporal/map_review/sam3_full_video/interior_half_review/consistent_tracks/centers_only.npz')
    with np.load(root/'map_review/sam3_full_video/interior_half_review/consistent_tracks/centers_only.npz') as z:
        use=np.isin(z['frame_ids'],frames);arrays={k:z[k][use] for k in z.files}
    (dest/relative).parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest/relative,**arrays)
    prefix=dest
    backend=dict(b,source_index=str(prefix/'source/index.json'),person_root=str(prefix/'persons'),vehicle_root=str(prefix/'vehicles'),temporal_output=str(prefix/'temporal'),pose_npz=str(prefix/'temporal/poses.npz'))
    (dest/'backend.json').write_text(json.dumps(backend,indent=2))
    fixture_profile=dict(c);fixture_profile.pop('backend');fixture_profile.pop('profile');fixture_profile['backend_config']='backend.json';fixture_profile['sam_source']=str(dependency_root()/'sam3');fixture_profile['semantic_runner']=str(REPO/'tools/autolabel/semantic_zgc/run_pipeline.py');fixture_profile['output']=str(directory('jobs')/'diagnostic');fixture_profile['review_output']=str(directory('jobs')/'diagnostic/review');fixture_profile['visual_sources']={name:str(prefix/'visual'/name) for name in c['visual_sources']}
    # Interpret all profile paths relative to this bundle, never the original project.
    import os
    for key in ['python','sam_source','semantic_runner','output','review_output']:fixture_profile[key]=os.path.relpath(fixture_profile[key],dest)
    fixture_profile['visual_sources']={name:os.path.relpath(REPO/folder,dest) for name,folder in fixture_profile['visual_sources'].items()}
    fixture_profile['diagnostic_only']=True
    (dest/'profile.json').write_text(json.dumps(fixture_profile,indent=2))
    manifest=dict(frames=sorted(frames),selection='Explicit historical diagnostic frames, not a random statistical sample',scope='Per-frame rendering, visual annotation inspection and geometry unit tests; not a complete temporal sequence or fresh semantic inference input',files={str(p.relative_to(dest)):hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.rglob('*') if p.is_file()})
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(dict(testdata=str(dest),frames=len(selected),files=len(manifest['files']))))


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--frames',type=int,nargs='+',default=[63,79,125,402,564,677,824,908]);p.add_argument('--output',type=Path,default=sample_path('testdata/zgc_review'));a=p.parse_args();export(a.config,a.frames,a.output)


if __name__=='__main__':main()
