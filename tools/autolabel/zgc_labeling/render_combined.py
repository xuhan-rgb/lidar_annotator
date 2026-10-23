"""Render saved vehicle 3D boxes and pedestrian XY labels together; no inference."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

TOOLS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(TOOLS/'pedestrian_instance_zgc'))
sys.path.insert(0,str(TOOLS/'wilddet3d_zgc'))
import render_outputs as pedestrian
from config import DATA, OUT as PERSON_ROOT, SOURCE
from label_geometry import boxes_to_corners_lidar
from projection_clipping import project_clipped_box_edges, _clip_segment_to_viewport

VEHICLE_ROOT=DATA/'wilddet3d_vehicle_camera_ray_distance030_20261014'
OUTPUT_ROOT=DATA/'combined_vehicle_pedestrian_labels_20261014'
VEHICLE_COLOR=(255,155,30)
PERSON_COLOR=(0,255,255)
EDGES=((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7))


def clip_bev_edge(a,b):
    segment=_clip_segment_to_viewport((a[0]-1260,a[1]-70),(b[0]-1260,b[1]-70),801,801)
    return None if segment is None else tuple((p[0]+1260,p[1]+70) for p in segment)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render_one(task):
    frame,row=task;name=f'{frame:06d}';dest=OUTPUT_ROOT/'visualization'
    pedestrian.COLORS=[PERSON_COLOR]
    canvas,(width,height)=pedestrian.render_frame(frame,PERSON_ROOT,mode='labels',return_canvas=True,label_text=False)
    vehicle_path=VEHICLE_ROOT/'refined'/f'{name}.npz'
    with np.load(vehicle_path) as z:
        boxes=z['box_3d'].astype(float);corners=z['corners_3d_lidar'].astype(float);seeds=z['seed_ids'].astype(int)
    np.testing.assert_allclose(boxes_to_corners_lidar(boxes),corners,atol=2e-6)
    with np.load(SOURCE/row['path']) as z:
        K=z['cam_intrinsic'].astype(float);E=z['cam_extrinsic'].astype(float)
        xyz=z['xyz'].astype(float);uv=z['pixels_uv'].astype(float)
    with np.load(VEHICLE_ROOT/'refined/support'/f'{name}.npz') as z:
        offsets=z['offsets'].astype(int);point_ids=z['point_indices'].astype(int)
        np.testing.assert_array_equal(z['seed_ids'],seeds)
    person_path=PERSON_ROOT/'frames'/f'{name}.npz'
    with np.load(person_path) as z:person_count=len(z['xy'])
    scale=min(1200/width,675/height);draw=ImageDraw.Draw(canvas)
    def image_xy(p):return (20+p[0]*scale,90+p[1]*scale)
    def bev_xy(x,y):return (1260+(20-y)*20,70+(20-x)*20)
    person_meta=json.loads((PERSON_ROOT/'metadata'/f'{name}.json').read_text())
    occupied=[]
    for item in person_meta['accepted']:
        iid=int(item['instance_id']);inst=person_meta['instances'][iid]
        pixel=image_xy(inst['sam_bbox_xyxy'][:2]);px,py=pixel[0],max(92,pixel[1]-23)
        text=f'P{iid} n={inst["point_support_count"]}';label_font=pedestrian.font(17)
        bounds=draw.textbbox((px,py),text,font=label_font)
        while py>92 and any(bounds[0]<b[2] and bounds[2]>b[0] and bounds[1]<b[3] and bounds[3]>b[1] for b in occupied):
            py=max(92,py-22);bounds=draw.textbbox((px,py),text,font=label_font)
        occupied.append(bounds)
        draw.text((px,py),text,font=label_font,fill=PERSON_COLOR,stroke_width=2,stroke_fill='black')
    for i,seed in enumerate(seeds):
        camera=corners[i]@E[:3,:3].T+E[:3,3]
        for _,a,b in project_clipped_box_edges(camera,EDGES,K,width,height,near=.1):
            draw.line((*image_xy(a),*image_xy(b)),fill=VEHICLE_COLOR,width=3)
        center=E[:3,:3]@boxes[i,:3]+E[:3,3]
        if center[2]>.1:
            projected=K@center;pixel=projected[:2]/projected[2]
            if np.isfinite(pixel).all() and 0<=pixel[0]<width and 0<=pixel[1]<height:
                px,py=image_xy(pixel);draw.ellipse((px-4,py-4,px+4,py+4),fill=VEHICLE_COLOR)
                draw.text((px+6,py),f'V{seed}',font=pedestrian.font(17),fill=VEHICLE_COLOR,stroke_width=2,stroke_fill='black')
        polygon=[bev_xy(*corners[i,j,:2]) for j in (0,1,5,4)]
        for a,b in zip(polygon,polygon[1:]+polygon[:1]):
            segment=clip_bev_edge(a,b)
            if segment is not None:draw.line((*segment[0],*segment[1]),fill=VEHICLE_COLOR,width=3)
        x,y=boxes[i,:2];px,py=bev_xy(x,y)
        if -20<x<20 and -20<y<20:
            draw.text((min(2020,px+8),max(75,min(845,py+5))),f'V{seed}',font=pedestrian.font(17),fill=VEHICLE_COLOR)
            length=max(.5,boxes[i,3]*.3);end=bev_xy(x+length*np.cos(boxes[i,6]),y+length*np.sin(boxes[i,6]))
            segment=clip_bev_edge((px,py),end)
            if segment is not None:draw.line((*segment[0],*segment[1]),fill=VEHICLE_COLOR,width=3)
            angle=np.arctan2(end[1]-py,end[0]-px)
            arrow=[end,(end[0]-10*np.cos(angle-.45),end[1]-10*np.sin(angle-.45)),(end[0]-10*np.cos(angle+.45),end[1]-10*np.sin(angle+.45))]
            if all(1260<=a<=2060 and 70<=b<=870 for a,b in arrow):draw.polygon(arrow,fill=VEHICLE_COLOR)
        ids=point_ids[offsets[i]:offsets[i+1]]
        for pid in ids:
            x,y=xyz[pid,:2]
            if np.isfinite([x,y]).all() and abs(x)<20 and abs(y)<20:
                px,py=bev_xy(x,y);draw.ellipse((px-2,py-2,px+2,py+2),fill=VEHICLE_COLOR)
            u,v=uv[pid]
            if np.isfinite([u,v]).all() and 0<=u<width and 0<=v<height:
                px,py=image_xy((u,v));draw.ellipse((px-2,py-2,px+2,py+2),fill=VEHICLE_COLOR)
    draw.rectangle((0,0,2080,65),fill=(20,23,28))
    draw.text((22,25),f'{name}  Saved labels · orange vehicles / cyan pedestrian XY',font=pedestrian.font(25),fill='white')
    draw.text((1260,30),f'BEV ±20 m · {len(boxes)} vehicles / {person_count} persons',font=pedestrian.font(24),fill='white')
    draw.rectangle((1260,939,2080,1020),fill=(20,23,28))
    draw.text((1270,945),'Orange: vehicle boxes + supports · cyan: person XY + supports',font=pedestrian.font(17),fill='white')
    draw.text((1270,975),'Gray: full LiDAR · geometry read from saved NPZ labels',font=pedestrian.font(18),fill='white')
    output=dest/f'{name}.png';canvas.save(output)
    with Image.open(output) as image:image.verify()
    hashes={str(path):digest(path) for path in [vehicle_path,person_path,VEHICLE_ROOT/'refined'/f'{name}.json',PERSON_ROOT/'metadata'/f'{name}.json']}
    record=dict(frame=name,vehicles=len(boxes),pedestrians=person_count,label_sha256=hashes,
                geometry_source='saved NPZ; vehicle box/corners and pedestrian XY, no inference',manual_gt=False,added_to_training=False)
    (dest/f'{name}_render.json').write_text(json.dumps(record,indent=2))
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames',nargs='+',type=int);parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args();dest=OUTPUT_ROOT/'visualization';dest.mkdir(parents=True,exist_ok=True)
    rows=json.loads((SOURCE/'index.json').read_text())['rows'];by_id={int(Path(r['path']).stem):r for r in rows}
    frames=sorted(by_id) if args.frames is None else sorted(set(args.frames))
    missing=set(frames)-by_id.keys()
    if missing:raise ValueError(f'Unknown frames: {sorted(missing)}')
    started=time.time();records=[];status=OUTPUT_ROOT/'render_status.json'
    status.write_text(json.dumps(dict(stage='running',total_frames=len(frames),workers=args.workers)))
    with ProcessPoolExecutor(max_workers=max(1,args.workers)) as pool:
        futures=[pool.submit(render_one,(frame,by_id[frame])) for frame in frames]
        for future in as_completed(futures):
            records.append(future.result())
            if len(records)%50==0 or len(records)==len(frames):
                status.write_text(json.dumps(dict(stage='running',completed_frames=len(records),total_frames=len(frames))))
                print(f'rendered {len(records)}/{len(frames)}',flush=True)
    pedestrian.write_gallery(sorted(dest.glob('*.png')),dest,visualization_dir='visualization',title='ZGC vehicle / pedestrian labels')
    # The reusable gallery writer uses ../visualization links, valid from this directory.
    report=dict(stage='complete',frames=len(records),vehicles=sum(r['vehicles'] for r in records),
                pedestrians=sum(r['pedestrians'] for r in records),workers=args.workers,elapsed_seconds=round(time.time()-started,3),
                vehicle_labels=str(VEHICLE_ROOT),pedestrian_labels=str(PERSON_ROOT),gallery=str(dest/'index.html'),
                labels_modified=False,manual_gt=False,added_to_training=False,pedestrian_confirmation='YOLO26x-only; frame 672 unconfirmed candidate ignored')
    status.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
