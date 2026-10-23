#!/usr/bin/env python3
"""Render label-only 2D/3D summaries from saved person-label NPZ/metadata."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw,ImageFont

from config import OUT,SOURCE,sha256
from geometry import NATIVE_TO_CUBE_ORDER
from process_instances import load_sam_frame
from resource_paths import font_path

COLORS=[(0,225,255),(255,95,75),(255,220,0),(115,245,100),(225,105,255),(255,155,30),(80,170,255),(245,245,245)]
EDGE_PAIRS=[(0,1),(1,5),(5,4),(4,0),(3,2),(2,6),(6,7),(7,3),(0,3),(1,2),(5,6),(4,7)]
FOOTPRINT=[0,1,5,4]


def font(size):
    return ImageFont.truetype(str(font_path('DejaVuSans.ttf')),size)


def render_frame(frame,root=OUT,dest=None,mode='labels',return_canvas=False,label_text=True):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'semantic_zgc'))
    from common import read_source_image
    rows=json.loads((SOURCE/'index.json').read_text())['rows'];by_frame={Path(r['path']).stem:r for r in rows}
    stem=f'{int(frame):06d}';row=by_frame[stem];image=read_source_image(row).convert('RGB')
    with np.load(SOURCE/row['path']) as source:E=np.asarray(source['cam_extrinsic'],float);K=np.asarray(source['cam_intrinsic'],float);source_xyz=np.asarray(source['xyz'],float)
    npz=None;corner_records=[]
    if mode=='labels':
        npz=np.load(root/'frames'/f'{stem}.npz');meta=json.loads((root/'metadata'/f'{stem}.json').read_text())
        accepted={int(x['instance_id']):x for x in meta['accepted']}
        masks,sam_meta=load_sam_frame(frame,OUT/'raw_sam_person')
        with np.load(SOURCE/row['path']) as source:
            source_uv=np.asarray(source['pixels_uv'],float)
        rgba=np.zeros((image.height,image.width,4),dtype=np.uint8)
        for iid,mask in enumerate(masks):
            rgba[mask,:3]=COLORS[iid%len(COLORS)];rgba[mask,3]=110 if iid in accepted else 35
        image=Image.alpha_composite(image.convert('RGBA'),Image.fromarray(rgba,'RGBA')).convert('RGB')
    elif mode=='native':
        raw=json.loads((OUT/'raw_wilddet_person'/'frames'/f'{stem}.json').read_text())
        accepted={}
        inv=np.linalg.inv(E)
        for item in raw['items']:
            if float(item['score'])<.6 or float(item['score_3d'])<.1:continue
            iid=int(item['wilddet_instance_id']);corners_cam=np.asarray(item['corners_cam'],float)[NATIVE_TO_CUBE_ORDER]
            corners_lidar=corners_cam@inv[:3,:3].T+inv[:3,3]
            center=corners_lidar.mean(0)
            accepted[iid]=dict(instance_id=iid,xy=center[:2].tolist(),confidence=float(item['score']),
                bbox_xyxy=item['bbox_xyxy'],box_3d_valid=True,score=float(item['score']),score_3d=float(item['score_3d']))
            corner_records.append((iid,corners_lidar,accepted[iid]))
    else:raise ValueError(mode)
    canvas=Image.new('RGB',(2080,1020),(20,23,28));draw=ImageDraw.Draw(canvas);f_title=font(26);f_label=font(19);f_tick=font(17)
    title=('WildDet3D person predictions · raw thresholds .6/.1' if mode=='native' else 'SAM masks · exclusive LiDAR points · saved XY')
    source_width,source_height=image.size
    image.thumbnail((1200,675),Image.Resampling.LANCZOS);scale_x=image.width/source_width;scale_y=image.height/source_height
    canvas.paste(image,(20,90));draw.text((22,25),f'{stem}  Camera · {title}',font=f_title,fill='white')
    imgdraw=ImageDraw.Draw(canvas)
    if mode=='labels':
        for iid,inst in enumerate(meta['instances']):
            color=COLORS[iid%len(COLORS)] if iid in accepted else (155,160,165)
            x0,y0,x1,y1=map(float,inst['sam_bbox_xyxy']);rect=(20+x0*scale_x,90+y0*scale_y,20+x1*scale_x,90+y1*scale_y)
            imgdraw.rectangle(rect,outline=color,width=2)
            if iid in accepted and label_text:imgdraw.text((rect[0],max(90,rect[1]-22-(iid%6)*22)),f'XY #{iid} n={len(inst["exclusive_support_points"])}',font=f_label,fill=color,stroke_width=2,stroke_fill=(0,0,0))
        offsets=np.asarray(npz['support_offsets'],int);point_ids=np.asarray(npz['support_indices'],int)
        for iid in range(len(meta['instances'])):
            a,b=map(int,offsets[iid:iid+2]);ids=point_ids[a:b]
            color=COLORS[iid%len(COLORS)] if iid in accepted else (130,135,140)
            ids=ids[(ids>=0)&(ids<len(source_uv))];uvp=np.floor(source_uv[ids]).astype(int)
            for x,y in uvp:
                if 0<=x<source_width and 0<=y<source_height:
                    px,py=20+x*scale_x,90+y*scale_y;imgdraw.ellipse((px-3,py-3,px+3,py+3),fill=color)
            if iid in accepted and b>a:
                point_xyz=np.median(np.asarray(npz['support_xyz'][a:b],float),axis=0)
                point_cam=E[:3,:3]@point_xyz+E[:3,3];uvh=K@point_cam
                if point_cam[2]>.1:
                    px,py=20+uvh[0]/uvh[2]*scale_x,90+uvh[1]/uvh[2]*scale_y
                    imgdraw.line((px-11,py,px+11,py),fill=(255,255,255),width=3);imgdraw.line((px,py-11,px,py+11),fill=(255,255,255),width=3)
    else:
        for iid,item in accepted.items():
            color=COLORS[iid%len(COLORS)];x0,y0,x1,y1=map(float,item['bbox_xyxy'])
            rect=(20+x0*scale_x,90+y0*scale_y,20+x1*scale_x,90+y1*scale_y)
            imgdraw.rectangle(rect,outline=color,width=3)
            imgdraw.text((rect[0],max(90,rect[1]-24-(iid%4)*23)),f'3D id={iid} score={item["confidence"]:.2f}',font=f_label,fill=color,stroke_width=2,stroke_fill=(0,0,0))
    # Project the NPZ-stored 3D cuboid corners into the camera image.
    id_all=[x[0] for x in corner_records]
    for iid,corners,item in corner_records:
        color=COLORS[int(iid)%len(COLORS)]
        camera=corners@E[:3,:3].T+E[:3,3];uvw=camera@K.T
        uv=uvw[:,:2]/np.maximum(uvw[:,2:3],1e-8);valid=camera[:,2]>.1
        for a,b in EDGE_PAIRS:
            if valid[a] and valid[b]:
                p=(20+uv[a,0]*scale_x,90+uv[a,1]*scale_y);q=(20+uv[b,0]*scale_x,90+uv[b,1]*scale_y)
                imgdraw.line((*p,*q),fill=color,width=4)
    # BEV is X-up, +Y-left; coordinates are directly the exported box coordinates/corners.
    bx0,by0=1260,70;side=800;bd=ImageDraw.Draw(canvas);f_bev=font(22)
    def bev_xy(x,y):return (bx0+(20-float(y))*side/40,by0+(20-float(x))*side/40)
    for value in range(-20,21,5):
        px=bx0+(20-value)*side/40;py=by0+(20-value)*side/40
        bd.line((px,by0,px,by0+side),fill=(50,55,62),width=1);bd.line((bx0,py,bx0+side,py),fill=(50,55,62),width=1)
        bd.text((px-14,by0+side+8),str(value),font=f_tick,fill=(235,235,235))
        bd.text((bx0-39,py-10),str(value),font=f_tick,fill=(235,235,235))
    bd.rectangle((bx0,by0,bx0+side,by0+side),outline=(170,175,180),width=2)
    # The horizontal world axis shown is Y; positive Y is on the left.
    bd.text((bx0+side//2-45,by0+side+37),'Y (m)  +Y left',font=f_label,fill='white')
    bd.text((bx0+8,by0+8),'X (m)  +X up',font=f_label,fill='white')
    background=source_xyz[np.isfinite(source_xyz).all(axis=1)&(np.abs(source_xyz[:,0])<20)&(np.abs(source_xyz[:,1])<20)]
    for x,y in background[:,:2]:
        px,py=bev_xy(x,y);bd.ellipse((px-1,py-1,px+1,py+1),fill=(95,100,105))
    for iid,corners,item in corner_records:
        color=COLORS[int(iid)%len(COLORS)];poly=[bev_xy(corners[i,0],corners[i,1]) for i in FOOTPRINT]
        for a,b in zip(poly,poly[1:]+poly[:1]):bd.line((*a,*b),fill=color,width=4)
        x,y=np.asarray(accepted[int(iid)]['xy']);px,py=bev_xy(x,y);bd.ellipse((px-5,py-5,px+5,py+5),fill=color)
    if mode=='labels':
        offsets=np.asarray(npz['support_offsets'],int);support_xyz=np.asarray(npz['support_xyz'],float)
        for iid in range(len(meta['instances'])):
            a,b=map(int,offsets[iid:iid+2]);color=COLORS[iid%len(COLORS)] if iid in accepted else (95,100,105)
            for x,y in support_xyz[a:b,:2]:
                px,py=bev_xy(x,y);bd.ellipse((px-2,py-2,px+2,py+2),fill=color)
    box_ids=set(map(int,id_all))
    for iid,item in accepted.items():
        if iid in box_ids:continue
        x,y=item['xy'];px,py=bev_xy(x,y);color=COLORS[iid%len(COLORS)]
        bd.ellipse((px-7,py-7,px+7,py+7),outline=color,width=4);bd.line((px-5,py-5,px+5,py+5),fill=color,width=2);bd.line((px+5,py-5,px-5,py+5),fill=color,width=2)
        if mode=='labels':
            bd.line((px-11,py,px+11,py),fill=(255,255,255),width=2);bd.line((px,py-11,px,py+11),fill=(255,255,255),width=2)
    # Fixed-width legend at right of the BEV axes, with explicit distinction for XY-only.
    legend_y=by0+side+80
    if mode=='native':
        bd.line((bx0+10,legend_y+8,bx0+50,legend_y+8),fill=COLORS[0],width=4);bd.text((bx0+60,legend_y-4),'Raw Mono3D box footprint',font=f_label,fill='white')
    else:
        bd.ellipse((bx0+12,legend_y+1,bx0+26,legend_y+15),fill=COLORS[0]);bd.text((bx0+60,legend_y-4),'Saved XY coordinatewise median',font=f_label,fill='white')
        bd.ellipse((bx0+15,legend_y+35,bx0+23,legend_y+43),fill=COLORS[0]);bd.text((bx0+60,legend_y+28),'Gray: full LiDAR · color: instance points',font=f_label,fill='white')
    bd.text((bx0,30),f'BEV ±20 m  ·  {len(id_all)} 3D'+(f' / {len(accepted)-len(id_all)} saved XY' if mode=='labels' else ' raw boxes'),font=f_title,fill='white')
    if return_canvas:
        if npz is not None:npz.close()
        return canvas,(source_width,source_height)
    dest=Path(dest) if dest else Path(root)/'visualizations';dest.mkdir(parents=True,exist_ok=True)
    path=dest/f'{stem}.png';canvas.save(path)
    if npz is not None:npz.close()
    return path


def write_gallery(paths,dest,visualization_dir='visualizations',title='Person labels'):
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=True);pages=(len(paths)+49)//50
    for page in range(pages):
        chunk=paths[page*50:(page+1)*50];items=''.join(f'<a href="../{visualization_dir}/{p.name}"><img loading="lazy" src="../{visualization_dir}/{p.name}"><span>{p.stem}</span></a>' for p in chunk)
        nav=' '.join(f'<a href="page_{i+1:02d}.html">{i+1}</a>' for i in range(pages))
        (dest/f'page_{page+1:02d}.html').write_text(f'<!doctype html><meta charset="utf-8"><title>{title} {page+1}</title><style>body{{font:16px sans-serif;background:#17191c;color:#eee;margin:20px}}nav{{padding:12px;position:sticky;top:0;background:#17191c}}main{{display:grid;grid-template-columns:repeat(2,minmax(400px,1fr));gap:12px}}a{{color:#fff}}img{{width:100%;display:block}}span{{display:block;padding:5px}}</style><nav>{nav}</nav><main>{items}</main>')
    (dest/'index.html').write_text('<meta http-equiv="refresh" content="0;url=page_01.html">')


def main():
    parser=argparse.ArgumentParser(description=__doc__);g=parser.add_mutually_exclusive_group(required=True);g.add_argument('--all',action='store_true');g.add_argument('--frames',type=int,nargs='+');g.add_argument('--changed',action='store_true')
    parser.add_argument('--root',type=Path,default=OUT);parser.add_argument('--output',type=Path);parser.add_argument('--mode',choices=['labels','native'],default='labels');args=parser.parse_args()
    root=args.root.resolve()
    default_dir='visualizations' if args.mode=='labels' else 'raw_wilddet_visualizations'
    dest=args.output.resolve() if args.output else root/default_dir;dest.mkdir(parents=True,exist_ok=True)
    if args.all:frames=sorted(int(p.stem) for p in (root/'frames').glob('*.npz'))
    elif args.changed:
        change=json.loads((root/'changed_frames.json').read_text());frames=sorted(int(x['frame']) for x in change['frames'])
    else:frames=sorted(set(args.frames))
    if args.all and len(frames)!=971:raise RuntimeError(f'Expected 971 exported frames, found {len(frames)}')
    status=root/('render_status.json' if args.mode=='labels' else 'native_render_status.json');status.write_text(json.dumps(dict(stage='running',pid=__import__('os').getpid(),frames=len(frames),mode=args.mode,started=time.time()),indent=2))
    paths=[]
    for i,frame in enumerate(frames,1):
        path=render_frame(frame,root,dest,args.mode);Image.open(path).verify();paths.append(path)
        if i%50==0 or i==len(frames):status.write_text(json.dumps(dict(stage='running',pid=__import__('os').getpid(),completed_frames=i,total_frames=len(frames),updated=time.time()),indent=2));print(f'rendered {i}/{len(frames)}',flush=True)
    gallery_dir='gallery' if args.mode=='labels' else 'raw_wilddet_gallery'
    if not args.all:
        all_existing=sorted(dest.glob('*.png'))
        gallery_paths=all_existing if len(all_existing)==971 else paths
    else:gallery_paths=paths
    write_gallery(gallery_paths,root/gallery_dir,default_dir)
    status.write_text(json.dumps(dict(stage='complete',completed_frames=len(paths),total_frames=len(frames),
        mode=args.mode,rendered_pngs_this_run=len(paths),gallery_pngs=len(gallery_paths),decoded_pngs=len(paths),
        visualizations=str(dest),gallery=str(root/gallery_dir/'index.html'),
        random_sampling=False,finished=time.time()),indent=2))


if __name__=='__main__':main()
