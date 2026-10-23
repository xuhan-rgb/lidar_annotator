"""Read the index.lmdb + NPZ shard recording format without modifying the source."""
import argparse,hashlib,io,json,signal,time
from pathlib import Path
import lmdb
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from .display_range import point_mask
from tools.autolabel.resource_paths import font_path
import msgpack


def frame_records(source):
    source=Path(source).expanduser().resolve()
    if not (source/'index.lmdb/data.mdb').is_file():raise ValueError('请选择包含 index.lmdb 和 shards 的采集根目录')
    with lmdb.open(str(source/'index.lmdb'),readonly=True,lock=False,create=False) as env:
        with env.begin() as txn:
            cursor=txn.cursor();records=[]
            if cursor.set_range(b'frame_index/'):
                for key,value in cursor:
                    if not key.startswith(b'frame_index/'):break
                    record=msgpack.unpackb(value,raw=False)
                    for field in ['global_frame_id','shard_id','lmdb_key','timestamp_ns']:
                        if field not in record:raise ValueError(f'LMDB 索引缺少字段 {field}')
                    records.append(record)
    records.sort(key=lambda r:int(r['global_frame_id']))
    ids=[int(r['global_frame_id']) for r in records]
    if not records or len(set(ids))!=len(ids):raise ValueError('LMDB 帧索引为空或有重复帧号')
    return source,records


def decode_frame(payload):
    with np.load(io.BytesIO(payload),allow_pickle=False) as archive:raw={k:archive[k] for k in archive.files}
    points=np.asarray(raw['lidar_points'])
    if points.ndim!=2 or points.shape[1]<3:raise ValueError('lidar_points 必须为 [点数,至少3列]')
    K=np.asarray(raw['cam_intrinsic'],float);E=np.asarray(raw['cam_extrinsic'],float)
    if K.shape!=(3,3) or E.shape!=(4,4) or not np.isfinite(K).all() or not np.isfinite(E).all():raise ValueError('相机标定矩阵维度或数值无效')
    jpeg=raw['image_jpeg_bytes'].tobytes()
    image=Image.open(io.BytesIO(jpeg)).convert('RGB')
    xyz=points[:,:3].astype(np.float32);camera=xyz.astype(float)@E[:3,:3].T+E[:3,3];homogeneous=camera@K.T
    uv=np.full((len(xyz),2),np.nan,np.float32);positive=np.isfinite(camera).all(1)&(camera[:,2]>.1)
    uv[positive]=homogeneous[positive,:2]/homogeneous[positive,2:]
    in_image=positive&(uv[:,0]>=0)&(uv[:,0]<image.width)&(uv[:,1]>=0)&(uv[:,1]<image.height)
    arrays=dict(xyz=xyz,lidar_points=points,cam_intrinsic=K,cam_extrinsic=E,pixels_uv=uv,camera_depth=camera[:,2].astype(np.float32),in_image=in_image,lidar_timestamp_ns=np.asarray(raw['lidar_timestamp_ns'],np.int64))
    for key in ['image_timestamp_ns','image_time_offset_ms','pose_world','robot_state','frame_id']:
        if key in raw:arrays[key]=raw[key]
    return arrays,jpeg,image,raw.get('semantic_class_png_bytes')


def preview_frame(image,arrays,destination,frame):
    canvas=Image.new('RGB',(2080,1020),(20,23,28));image=image.copy();original=image.size;image.thumbnail((1200,675));canvas.paste(image,(20,90));draw=ImageDraw.Draw(canvas);font=ImageFont.truetype(str(font_path('DejaVuSans.ttf')),21)
    def bev(point):return (1260+(20-point[1])*20,70+(20-point[0])*20)
    for v in range(-20,21,5):
        x=1260+(20-v)*20;y=70+(20-v)*20;draw.line((x,70,x,870),fill=(50,55,62));draw.line((1260,y,2060,y),fill=(50,55,62));draw.text((x-12,880),str(v),font=font,fill='white');draw.text((1210,y-10),str(v),font=font,fill='white')
    from matplotlib import colormaps
    cmap=colormaps['turbo_r']
    def colors(values):return (cmap(np.clip(values/40.,0,1))[:,:3]*255).astype(np.uint8)
    xyz=arrays['xyz'];valid=point_mask(xyz)
    for p,color in zip(xyz[valid],colors(np.linalg.norm(xyz[valid],axis=1))):draw.point(bev(p),fill=tuple(map(int,color)))
    depth=arrays['camera_depth'];projected=valid&arrays['in_image']&np.isfinite(depth)&(depth>0)&np.isfinite(arrays['pixels_uv']).all(1)
    for (u,v),color in zip(arrays['pixels_uv'][projected],colors(depth[projected])):
        x=20+u*image.width/original[0];y=90+v*image.height/original[1];draw.ellipse((x-1,y-1,x+1,y+1),fill=tuple(map(int,color)))
    for x,y,title in ((22,880,'Projection: camera Z depth (m)'),(1260,960,'BEV: distance to LiDAR origin (m)')):
        draw.text((x,y-30),title,font=font,fill='white')
        for offset,color in enumerate(colors(np.linspace(0,40,400))):draw.line((x+offset,y,x+offset,y+14),fill=tuple(map(int,color)))
        for offset,label in ((0,'0 (near)'),(185,'20'),(350,'40+ (far)')):draw.text((x+offset,y+18),label,font=font,fill='white')
    draw.rectangle((1260,70,2060,870),outline=(170,175,180));draw.text((22,25),f'{frame:06d}  LMDB raw image + projected LiDAR',font=font,fill='white');draw.text((1260,25),'Raw BEV +/-20m | X up, Y left',font=font,fill='white');draw.text((22,810),'Raw input preview: no generated semantic / person / vehicle labels',font=font,fill='white');canvas.save(destination)


def read_lmdb(source,output,frames=None,preview=False,dry_run=False,kitti_output=None):
    source,records=frame_records(source);wanted=set(frames) if frames is not None else None
    chosen=[r for r in records if wanted is None or int(r['global_frame_id']) in wanted]
    if wanted is not None and wanted-{int(r['global_frame_id']) for r in records}:raise ValueError('指定帧号不在 LMDB 索引中')
    output=Path(output).expanduser().absolute()
    if output==source or source in output.parents:raise ValueError('输出目录必须在原始 LMDB 目录之外')
    if dry_run:return dict(status='dry_run',format='indexed_npz_lmdb_v1',source=str(source),total_frames=len(records),selected_frames=len(chosen),output=str(output),preview=preview,source_readonly=True)
    output.mkdir(parents=True,exist_ok=True)
    for folder in ['frames','images','source_semantics','review/visualization']:(output/folder).mkdir(parents=True,exist_ok=True)
    owner=output/'lmdb_source.json';identity=dict(source=str(source),schema='indexed_npz_lmdb_v1')
    if owner.exists() and json.loads(owner.read_text())!=identity:raise ValueError('输出目录属于另一个 LMDB 数据源，请使用新目录')
    owner.write_text(json.dumps(identity,indent=2));rows=[];poses=[];times=[];ids=[];started=time.monotonic();current=None;env=None
    (output/'status.json').write_text(json.dumps(dict(status='running',completed=0,total=len(chosen))))
    try:
        for ordinal,record in enumerate(chosen,1):
            frame=int(record['global_frame_id']);shard=source/'shards'/f"shard_{int(record['shard_id']):06d}.lmdb"
            if current!=shard:
                if env is not None:env.close()
                env=lmdb.open(str(shard),readonly=True,lock=False,create=False);current=shard
            with env.begin() as txn:payload=txn.get(record['lmdb_key'].encode())
            if payload is None:raise ValueError(f'分片缺少帧 {frame}')
            digest=hashlib.sha256(payload).hexdigest()
            if record.get('payload_sha256') and digest!=record['payload_sha256']:raise ValueError(f'帧 {frame} 的 LMDB 载荷哈希不匹配')
            arrays,jpeg,image,semantic=decode_frame(payload);stem=f'{frame:06d}';np.savez_compressed(output/'frames'/f'{stem}.npz',**arrays);(output/'images'/f'{stem}.jpg').write_bytes(jpeg)
            if semantic is not None:(output/'source_semantics'/f'{stem}.png').write_bytes(semantic.tobytes())
            if preview:preview_frame(image,arrays,output/'review/visualization'/f'{stem}.png',frame)
            rows.append(dict(frame_id=frame,path=f'frames/{stem}.npz',image_path=str(output/'images'/f'{stem}.jpg'),original_shard=str(shard),original_lmdb_key=record['lmdb_key'],original_payload_sha256=digest,source_index_record=record))
            times.append(int(arrays['lidar_timestamp_ns']));ids.append(frame)
            pose=arrays.get('pose_world');poses.append(np.asarray(pose,float) if pose is not None and np.shape(pose)==(4,) else np.full(4,np.nan))
            if ordinal%25==0 or ordinal==len(chosen):
                status=dict(status='running',completed=ordinal,total=len(chosen));(output/'status.json').write_text(json.dumps(status));print(json.dumps(status),flush=True)
    except BaseException as exc:
        (output/'status.json').write_text(json.dumps(dict(status='cancelled' if isinstance(exc,KeyboardInterrupt) else 'failed',completed=len(rows),total=len(chosen),error=str(exc))));raise
    finally:
        if env is not None:env.close()
    (output/'index.json').write_text(json.dumps(dict(schema='lidar_raw_import_v1',rows=rows),ensure_ascii=False,indent=2));np.savez_compressed(output/'odometry.npz',poses=np.array(poses),timestamps=np.array(times,np.int64),frame_ids=np.array(ids,int))
    if preview:
        html='<!doctype html><meta charset="utf-8"><title>LMDB raw input review</title><style>body{background:#14171c;color:white}img{max-width:100%}</style><h1>LMDB 原始图像与点云预览</h1><p>未生成新的语义或实例标签。原有语义 PNG 仅保留原始类别编码。</p>'
        html+=''.join(f'<h2>{r["frame_id"]:06d}</h2><img loading="lazy" src="visualization/{r["frame_id"]:06d}.png">' for r in rows);(output/'review/index.html').write_text(html)
    report=dict(status='complete',source=str(source),source_readonly=True,total_recording_frames=len(records),imported_frames=len(rows),frame_ids=ids,output=str(output),preview=preview,elapsed_seconds=round(time.monotonic()-started,2),labels_generated=False,limits='Raw import and preview only. in_image is a projection/FOV mask, not an occlusion test. Original semantic PNG IDs are not remapped to annotation classes. This raw index is not a zgc_cached_v1 semantic/instance cache.')
    if kitti_output is not False:
        from .kitti_input import export_imported_kitti
        destination=Path(kitti_output) if kitti_output is not None else output/'kitti'
        def conversion_progress(done,total):
            status=dict(status='running',phase='kitti',completed=done,total=total)
            (output/'status.json').write_text(json.dumps(status))
            print(f'KITTI 转换：{done}/{total} 帧',flush=True)
        try:
            export_imported_kitti(output,destination,progress=conversion_progress)
            report['kitti_output']=str(destination);report['kitti_export_status']='complete'
            print('KITTI 导出目录：'+str(destination),flush=True)
        except KeyboardInterrupt:
            (output/'status.json').write_text(json.dumps(dict(status='cancelled',phase='kitti',error='KITTI conversion cancelled')))
            raise
        except (OSError,ValueError,KeyError) as exc:
            report['kitti_export_status']='failed';report['kitti_export_error']=str(exc)
            print('KITTI 导出未完成：'+str(exc),flush=True)
    report['elapsed_seconds']=round(time.monotonic()-started,2)
    (output/'report.json').write_text(json.dumps(report,indent=2));(output/'status.json').write_text(json.dumps(report));return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--frames',nargs='+',type=int);p.add_argument('--preview',action='store_true');p.add_argument('--dry-run',action='store_true');a=p.parse_args(argv)
    def cancel(signum,frame):raise KeyboardInterrupt('LMDB read cancelled')
    previous=signal.signal(signal.SIGTERM,cancel)
    try:print(json.dumps(read_lmdb(a.source,a.output,a.frames,a.preview,a.dry_run),ensure_ascii=False,indent=2))
    finally:signal.signal(signal.SIGTERM,previous)


if __name__=='__main__':main()
