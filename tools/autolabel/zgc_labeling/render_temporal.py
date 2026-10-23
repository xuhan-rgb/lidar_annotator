"""Render only saved temporal label geometry and selected supports for human review."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib,json
from pathlib import Path
import sys

import numpy as np
from PIL import Image,ImageDraw

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'pedestrian_instance_zgc'))
from render_outputs import font,write_gallery
from process_instances import load_sam_frame
from render_combined import EDGES,clip_bev_edge
from projection_clipping import project_clipped_box_edges


def render_one(task):
    frame,row,config,history=task;root=Path(config['temporal_output']);stem=f'{frame:06d}';source=Path(config['source_index']).parent
    sys.path.insert(0,str(Path(__file__).resolve().parent));from common import read_source_image
    image=read_source_image(row).convert('RGB');width,height=image.size
    meta=json.loads((root/'metadata'/f'{stem}.json').read_text());records=meta['observations']
    with np.load(source/row['path']) as z:xyz=z['xyz'].astype(float);uv=z['pixels_uv'].astype(float);E=z['cam_extrinsic'].astype(float);K=z['cam_intrinsic'].astype(float)
    with np.load(root/'frames'/f'{stem}.npz') as z:
        ps=z['person_instance_ids'].astype(int);p_xy=z['person_xy'].astype(float);vs=z['vehicle_seed_ids'].astype(int);boxes=z['vehicle_box_3d'].astype(float);corners=z['vehicle_corners_3d_lidar'].astype(float)
        point_ids=z['support_point_indices'].astype(int);offsets=z['support_offsets'].astype(int);pt=z['person_track_ids'].astype(int);vt=z['vehicle_track_ids'].astype(int)
        temporal_flags=z['vehicle_temporal_confirmed'].astype(bool) if 'vehicle_temporal_confirmed' in z else np.ones(len(vs),bool)
    pmap={o['instance_id']:o for o in records if o['kind']=='person'};vmap={o['seed_id']:o for o in records if o['kind']=='vehicle'}
    masks,_=load_sam_frame(frame,Path(config['person_root'])/'raw_sam_person');rgba=np.zeros((height,width,4),np.uint8)
    for iid in ps:rgba[masks[iid]]=(0,255,255,40)
    image=Image.alpha_composite(image.convert('RGBA'),Image.fromarray(rgba)).convert('RGB')
    image.thumbnail((1200,675));sx=image.width/width;sy=image.height/height
    canvas=Image.new('RGB',(2080,1020),(20,23,28));canvas.paste(image,(20,90));draw=ImageDraw.Draw(canvas)
    def pixel(p):return (20+p[0]*sx,90+p[1]*sy)
    def bev(x,y):return (1260+(20-y)*20,70+(20-x)*20)
    for v in range(-20,21,5):
        x=1260+(20-v)*20;y=70+(20-v)*20
        draw.line((x,70,x,870),fill=(50,55,62));draw.line((1260,y,2060,y),fill=(50,55,62))
        draw.text((x-12,878),str(v),font=font(17),fill='white');draw.text((1222,y-10),str(v),font=font(17),fill='white')
    valid=np.isfinite(xyz).all(1)&(np.abs(xyz[:,:2])<20).all(1)
    for x,y in xyz[valid,:2]:
        px,py=bev(x,y);draw.ellipse((px-1,py-1,px+1,py+1),fill=(95,100,105))
    draw.rectangle((1260,70,2060,870),outline=(170,175,180));draw.text((1270,80),'X (m) +X up',font=font(18),fill='white');draw.text((1540,910),'Y (m) +Y left',font=font(19),fill='white')
    # Ignored observations remain gray diagnostic rectangles, not positive positions.
    for o in records:
        if o['status'].startswith('ignore'):
            a=pixel(o['bbox'][:2]);b=pixel(o['bbox'][2:]);draw.rectangle((*a,*b),outline=(145,150,155),width=1)
    for i,iid in enumerate(ps):
        o=pmap[iid];color=(0,255,255);a=pixel(o['bbox'][:2]);b=pixel(o['bbox'][2:]);draw.rectangle((*a,*b),outline=color,width=2)
        draw.text((a[0],max(91,a[1]-20)),f'P{iid}/T{pt[i]}',font=font(17),fill=color,stroke_width=2,stroke_fill='black')
        point=np.array(o['point']);point[:2]=p_xy[i];cam=E[:3,:3]@point+E[:3,3]
        if cam[2]>.1:
            h=K@cam;px,py=pixel(h[:2]/h[2]);draw.line((px-8,py,px+8,py),fill='white',width=2);draw.line((px,py-8,px,py+8),fill='white',width=2)
        px,py=bev(*p_xy[i]);draw.line((px-8,py,px+8,py),fill=color,width=3);draw.line((px,py-8,px,py+8),fill=color,width=3)
        draw.text((min(px+8,1970),min(py+8,845)),f'P/T{pt[i]}',font=font(16),fill=color)
        path=history.get((int(meta['segment']),int(pt[i])),[])
        prior=[o2 for o2 in path if 0<=o['time']-o2['time']<=1.]
        T=np.array(config['frame_transforms'][str(frame)]);inverse=np.linalg.inv(T)
        points=[bev(*(inverse[:3,:3]@np.asarray(q['world'])+inverse[:3,3])[:2]) for q in prior]
        for a,b in zip(points,points[1:]):
            edge=clip_bev_edge(a,b)
            if edge:draw.line((*edge[0],*edge[1]),fill=(0,130,150),width=2)
    for i,seed in enumerate(vs):
        color=(255,155,30);cam=corners[i]@E[:3,:3].T+E[:3,3]
        for _,a,b in project_clipped_box_edges(cam,EDGES,K,width,height):draw.line((*pixel(a),*pixel(b)),fill=color,width=3)
        polygon=[bev(*corners[i,j,:2]) for j in [0,1,5,4]]
        for a,b in zip(polygon,polygon[1:]+polygon[:1]):
            edge=clip_bev_edge(a,b)
            if edge:draw.line((*edge[0],*edge[1]),fill=color,width=3)
        x,y=boxes[i,:2];px,py=bev(x,y);length=max(.5,boxes[i,3]*.3);end=bev(x+length*np.cos(boxes[i,6]),y+length*np.sin(boxes[i,6]))
        edge=clip_bev_edge((px,py),end)
        if edge:draw.line((*edge[0],*edge[1]),fill=color,width=3)
        angle=np.arctan2(end[1]-py,end[0]-px)
        arrow=[end,(end[0]-10*np.cos(angle-.45),end[1]-10*np.sin(angle-.45)),(end[0]-10*np.cos(angle+.45),end[1]-10*np.sin(angle+.45))]
        if all(1260<=a<=2060 and 70<=b<=870 for a,b in arrow):draw.polygon(arrow,fill=color)
        draw.text((max(1265,min(1990,px+8)),max(75,min(845,py+8))),(f'V/T{vt[i]}' if temporal_flags[i] else 'V/S'),font=font(16),fill=color)
    for i in range(len(ps)+len(vs)):
        color=(0,255,255) if i<len(ps) else (255,155,30)
        for pid in point_ids[offsets[i]:offsets[i+1]]:
            x,y=xyz[pid,:2]
            if abs(x)<20 and abs(y)<20:
                px,py=bev(x,y);draw.ellipse((px-2,py-2,px+2,py+2),fill=color)
            u,v=uv[pid]
            if 0<=u<width and 0<=v<height:
                px,py=pixel((u,v));draw.ellipse((px-2,py-2,px+2,py+2),fill=color)
    draw.text((22,25),f'{stem}  Temporal REVIEW · saved NPZ labels · segment {meta["segment"]}',font=font(25),fill='white')
    draw.text((1260,30),f'BEV ±20m · {len(vs)} vehicles [{int(temporal_flags.sum())}T/{int((~temporal_flags).sum())}S] / {len(ps)} persons',font=font(24),fill='white')
    draw.text((1260,950),'Orange vehicles: T = temporal / S = single frame · cyan persons',font=font(18),fill='white')
    draw.text((1260,982),'Gray boxes: ignored observations · gray dots: raw LiDAR',font=font(18),fill='white')
    dest=root/'visualization';canvas.save(dest/f'{stem}.png')
    with Image.open(dest/f'{stem}.png') as check:check.verify()
    record=dict(frame=stem,label_npz_sha256=hashlib.sha256((root/'frames'/f'{stem}.npz').read_bytes()).hexdigest(),person_count=len(ps),vehicle_count=len(vs),geometry_source='temporal NPZ',manual_gt=False,added_to_training=False)
    (dest/f'{stem}_render.json').write_text(json.dumps(record,indent=2));return record


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--workers',type=int,default=4);p.add_argument('--frames',type=int,nargs='+');a=p.parse_args()
    config=json.loads(a.config.read_text());source=Path(config['source_index']);rows=json.loads(source.read_text())['rows']
    root=Path(config['temporal_output']);dest=root/'visualization';dest.mkdir(parents=True,exist_ok=True)
    with np.load(config['pose_npz']) as z:config['frame_transforms']={str(int(Path(row['path']).stem)):T.tolist() for row,T in zip(rows,z['transforms'])}
    history={}
    for path in sorted((root/'metadata').glob('*.json')):
        d=json.loads(path.read_text())
        for o in d['observations']:
            if o['kind']=='person' and o['status']=='accepted_temporal':history.setdefault((d['segment'],o['track_id']),[]).append(o)
    tasks=[(int(Path(row['path']).stem),row,config,history) for row in rows if a.frames is None or int(Path(row['path']).stem) in a.frames]
    records=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(render_one,t) for t in tasks]
        for future in as_completed(futures):
            records.append(future.result())
            if len(records)%100==0:print(f'rendered {len(records)}/{len(tasks)}',flush=True)
    write_gallery(sorted(dest.glob('*.png')),dest,'visualization',title='Temporal labels review')
    report=dict(status='complete',frames=len(records),person_xy=sum(r['person_count'] for r in records),vehicle_boxes=sum(r['vehicle_count'] for r in records),geometry_source='saved temporal NPZ',manual_gt=False,added_to_training=False)
    (root/'render_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))


if __name__=='__main__':main()
