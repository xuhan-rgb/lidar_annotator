"""Export indexed LMDB recordings and import portable KITTI odometry folders."""
import argparse,hashlib,io,json,shutil,tempfile
from decimal import Decimal,InvalidOperation,ROUND_HALF_EVEN
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

from .lmdb_input import decode_frame,frame_records as lmdb_frame_records


def _seconds_to_ns(value):
    try:return int((Decimal(value)*Decimal(1_000_000_000)).to_integral_value(rounding=ROUND_HALF_EVEN))
    except (InvalidOperation,ValueError):return None


def _record_payload(source,record,envs):
    shard=source/'shards'/f"shard_{int(record['shard_id']):06d}.lmdb"
    if shard not in envs:
        import lmdb
        envs[shard]=lmdb.open(str(shard),readonly=True,lock=False,create=False)
    with envs[shard].begin() as txn:payload=txn.get(record['lmdb_key'].encode())
    if payload is None:raise ValueError(f"分片缺少帧 {record['global_frame_id']}")
    digest=hashlib.sha256(payload).hexdigest()
    if record.get('payload_sha256') and digest!=record['payload_sha256']:raise ValueError(f"帧 {record['global_frame_id']} 的 LMDB 载荷哈希不匹配")
    return payload,digest


def export_lmdb_kitti(source,output,count=10):
    source,records=lmdb_frame_records(source)
    if count<1:raise ValueError('count 必须为正数')
    chosen=records[:count]
    output=Path(output).expanduser().absolute()
    if output.exists() and any(output.iterdir()):raise ValueError('输出目录必须不存在或为空')
    if output==source or source in output.parents:raise ValueError('输出目录必须在原始 LMDB 目录之外')
    envs={};decoded=[]
    try:
        for r in chosen:
            payload,digest=_record_payload(source,r,envs);arrays,jpeg,image,_=decode_frame(payload)
            pose=np.asarray(arrays.get('pose_world',np.full(4,np.nan)),float)
            if pose.shape!=(4,) or (np.isfinite(pose).any() and not np.isfinite(pose).all()):raise ValueError(f"帧 {r['global_frame_id']} 的 pose_world 无效")
            K=np.asarray(arrays['cam_intrinsic'],float);E=np.asarray(arrays['cam_extrinsic'],float)
            if abs(np.linalg.det(K))<1e-12 or not np.allclose(E[3],[0,0,0,1]) or abs(np.linalg.det(E[:3,:3]))<1e-9:raise ValueError('相机标定矩阵无效')
            decoded.append((r,arrays,jpeg,image,digest))
    finally:
        for env in envs.values():env.close()
    K0=decoded[0][1]['cam_intrinsic'];E0=decoded[0][1]['cam_extrinsic'];fixed_calib=all(np.allclose(a['cam_intrinsic'],K0) and np.allclose(a['cam_extrinsic'],E0) for _,a,*_ in decoded)
    output.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix=f'.{output.name}.',dir=output.parent))
    try:
        (stage/'image_2').mkdir();(stage/'velodyne').mkdir()
        first_ns=int(decoded[0][1]['lidar_timestamp_ns']);rows=[];poses=[]
        for local,(r,a,jpeg,image,digest) in enumerate(decoded):
            stem=f'{local:06d}';image.save(stage/'image_2'/f'{stem}.png',format='PNG')
            # KITTI expects Nx4. Preserve original columns in metadata.npz.
            pts=np.asarray(a['lidar_points']);intensity=pts[:,3] if pts.shape[1]>=4 else np.zeros(len(pts),np.float32)
            np.column_stack((pts[:,:3],intensity)).astype('<f4').tofile(stage/'velodyne'/f'{stem}.bin')
            pose=np.asarray(a.get('pose_world',np.full(4,np.nan)),float)
            if np.isfinite(pose).all():
                T=np.eye(4);x,y,z,yaw=pose;c,s=np.cos(yaw),np.sin(yaw);T[:3,:3]=[[c,-s,0],[s,c,0],[0,0,1]];T[:3,3]=[x,y,z]
            else:T=np.full((4,4),np.nan)
            poses.append(T)
            rows.append(dict(local_frame_id=local,global_frame_id=int(r['global_frame_id']),timestamp_ns=int(a['lidar_timestamp_ns']),image_timestamp_ns=int(np.asarray(a.get('image_timestamp_ns',a['lidar_timestamp_ns']))),source_record=r,payload_sha256=digest,point_columns=int(pts.shape[1])))
        # KITTI projection is P2=K[I|0], and Tr is the original LiDAR-to-camera transform.
        calib_dir=stage/'calib';
        if not fixed_calib:calib_dir.mkdir()
        for local,(_,a,*_) in enumerate(decoded):
            if fixed_calib and local:continue
            K=a['cam_intrinsic'];E=a['cam_extrinsic'];body='P2: '+' '.join(map(str,np.c_[K,np.zeros(3)].ravel()))+'\nTr: '+' '.join(map(str,E[:3].ravel()))+'\n'
            (stage/('calib.txt' if fixed_calib else f'calib/{local:06d}.txt')).write_text(body)
        (stage/'times.txt').write_text(''.join(f'{(int(row["timestamp_ns"])-first_ns)/1e9:.9f}\n' for row in rows))
        E=np.eye(4);E[:3]=E0[:3]
        if np.isfinite(np.asarray([a.get('pose_world',np.full(4,np.nan)) for _,a,*_ in decoded],float)).all():
            camera_poses=[E0@np.linalg.inv(poses[0])@p@np.linalg.inv(E0) for p in poses]
            (stage/'poses.txt').write_text(''.join(' '.join(f'{v:.12g}' for v in T[:3].ravel())+'\n' for T in camera_poses))
        pose_path=stage/'poses.txt';poses_hash=hashlib.sha256(pose_path.read_bytes()).hexdigest() if pose_path.is_file() else ''
        times_hash=hashlib.sha256((stage/'times.txt').read_bytes()).hexdigest()
        metadata={'provenance':np.array(json.dumps(dict(format='openpcdet_lmdb_kitti_v1',source=str(source),records=rows,fields={'points_i':'original full point rows; columns 0:3 xyz, column 3 intensity when present','jpeg_i':'exact source JPEG bytes','lidar_timestamp_ns':'authoritative signed nanoseconds','image_timestamp_ns':'authoritative image nanoseconds','source_pose_world':'source [x,y,z,yaw]'},status='converted; no labels or ground truth exported'),ensure_ascii=False)), 'global_frame_ids':np.array([x['global_frame_id'] for x in rows],np.int64), 'local_frame_ids':np.array([x['local_frame_id'] for x in rows],np.int64), 'lidar_timestamp_ns':np.array([x['timestamp_ns'] for x in rows],np.int64), 'image_timestamp_ns':np.array([x['image_timestamp_ns'] for x in rows],np.int64), 'source_pose_world':np.array([np.asarray(a.get('pose_world',np.full(4,np.nan)),float) for _,a,*_ in decoded]),'poses_txt_sha256':np.array(poses_hash),'times_txt_sha256':np.array(times_hash), 'cam_intrinsic':K0, 'cam_extrinsic':E0}
        for i,(_,a,jpeg,image,_) in enumerate(decoded):
            metadata[f'points_{i}']=a['lidar_points'];metadata[f'jpeg_{i}']=np.frombuffer(jpeg,np.uint8)
        np.savez_compressed(stage/'metadata.npz',**metadata)
        if output.exists():output.rmdir()
        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage,ignore_errors=True);raise
    return dict(status='complete',output=str(output),frame_ids=[x['global_frame_id'] for x in rows],exported_frames=len(rows),input_format='indexed_npz_lmdb_v1')


def camera_channels(source):
    source=Path(source).expanduser().resolve()
    return sorted(p.name for p in source.glob('image_*') if p.is_dir() and p.name.removeprefix('image_').isdigit() and any(x.is_file() and x.suffix.lower() in ('.png','.jpg','.jpeg') for x in p.iterdir()))


def frame_records(source,camera=None):
    source=Path(source).expanduser().resolve()
    channels=camera_channels(source)
    if not (source/'velodyne').is_dir() or not channels:raise ValueError('需要 KITTI sequence 根目录，含 image_n 和 velodyne')
    camera=camera or ('image_2' if 'image_2' in channels else channels[0])
    if camera not in channels:raise ValueError(f'KITTI sequence 不存在相机通道 {camera}')
    images={int(p.stem):p for p in (source/camera).glob('*') if p.suffix.lower() in ('.png','.jpg','.jpeg') and p.stem.isdigit()}
    points={int(p.stem):p for p in (source/'velodyne').glob('*.bin') if p.stem.isdigit()}
    ids=sorted(images.keys()&points.keys())
    if not ids:raise ValueError('KITTI sequence 没有匹配的图像与点云帧')
    times=(source/'times.txt').read_text().splitlines() if (source/'times.txt').is_file() else []
    exact=None;original_ids=None;metadata_local_ids=None
    if (source/'metadata.npz').is_file():
        with np.load(source/'metadata.npz',allow_pickle=False) as meta:
            if 'lidar_timestamp_ns' in meta and (source/'times.txt').is_file():
                saved=str(meta['times_txt_sha256'].item()) if 'times_txt_sha256' in meta else ''
                tp=source/'times.txt';current=hashlib.sha256(tp.read_bytes()).hexdigest() if tp.is_file() else ''
                if not saved or saved==current:exact=meta['lidar_timestamp_ns'].copy()
            if 'global_frame_ids' in meta:original_ids=meta['global_frame_ids'].copy()
            if 'local_frame_ids' in meta:metadata_local_ids=meta['local_frame_ids'].copy()
    records=[]
    for ordinal,i in enumerate(ids):
        mi=ordinal
        if metadata_local_ids is not None:
            matches=np.flatnonzero(metadata_local_ids==i)
            if len(matches):mi=int(matches[0])
        elif original_ids is not None:
            matches=np.flatnonzero(original_ids==i)
            if len(matches):mi=int(matches[0])
            elif i<len(original_ids):mi=i
        second_text=times[i].strip() if i<len(times) else None
        try:seconds=float(second_text) if second_text is not None else float('nan')
        except ValueError:seconds=float('nan')
        records.append(dict(global_frame_id=int(original_ids[mi]) if original_ids is not None and mi<len(original_ids) else i,local_frame_id=i,metadata_index=mi,image_path=images[i],points_path=points[i],timestamp_seconds=seconds,timestamp_ns=int(exact[mi]) if exact is not None and mi<len(exact) else _seconds_to_ns(second_text) if second_text is not None else None,camera=camera))
    return source,records


def source_image(source,frame,camera=None):
    _,records=frame_records(source,camera);row=next((r for r in records if r['global_frame_id']==int(frame)),None)
    if row is None:raise ValueError(f'KITTI sequence 缺少帧 {frame}')
    with Image.open(row['image_path']) as im:return im.convert('RGB')


def _write_png(imagebytes,path):
    with Image.open(io.BytesIO(imagebytes)) as image:image.convert('RGB').save(path,format='PNG')


def _missing_export_files(out,marker):
    expected=marker.get('output_manifest')
    if not isinstance(expected,dict) or not expected:raise ValueError('导出记录无有效文件清单，已保留现有数据')
    if any(Path(name).is_absolute() or '..' in Path(name).parts for name in expected):raise ValueError('导出记录包含非法文件路径')
    actual={str(p.relative_to(out)):p for p in out.rglob('*') if p.is_file() and p.name!='.lidar_annotator_kitti.json' and p.relative_to(out).parts[0]!='labels'}
    unexpected=set(actual)-set(expected)
    if unexpected:raise ValueError('导出目录含未记录的文件，已保留：'+', '.join(sorted(unexpected)[:3]))
    changed=[name for name,path in actual.items() if hashlib.sha256(path.read_bytes()).hexdigest()!=expected[name]]
    if changed:raise ValueError('原始导出文件已修改，禁止覆盖：'+', '.join(sorted(changed)[:3]))
    return set(expected)-set(actual)


def export_imported_kitti(raw_directory,output,progress=None):
    """Convert an existing lidar_raw_import_v1 directory without opening its LMDB source."""
    raw=Path(raw_directory).expanduser().resolve();index=json.loads((raw/'index.json').read_text())
    if index.get('schema')!='lidar_raw_import_v1':raise ValueError('输入不是 lidar_raw_import_v1')
    rows=index['rows'];out=Path(output).expanduser().absolute()
    owner=raw/'lmdb_source.json';origin=json.loads(owner.read_text())['source'] if owner.is_file() else str(raw)
    stable=[{k:v for k,v in row.items() if k not in ('image_path','original_shard')} for row in rows]
    signature=hashlib.sha256(json.dumps(stable,sort_keys=True).encode())
    for row in rows:
        for key in ('path','image_path'):
            p=Path(row[key]);p=(p if p.is_absolute() else raw/p).resolve();signature.update(str(p.relative_to(raw) if raw in p.parents else p).encode());signature.update(p.read_bytes())
    signature=signature.hexdigest();marker_name='.lidar_annotator_kitti.json';marker_path=out/marker_name
    repair=False;repaired_files=0
    if out.exists() and any(out.iterdir()):
        try:marker=json.loads(marker_path.read_text())
        except (OSError,ValueError):marker={}
        if marker.get('source')!=origin or marker.get('source_signature')!=signature:raise ValueError('导出记录与当前数据源或帧范围不一致，已保留现有文件；请使用新的导出目录')
        missing=_missing_export_files(out,marker)
        if not missing:return dict(status='complete',output=str(out),exported_frames=len(rows),frame_ids=[r['frame_id'] for r in rows],input_format='lidar_raw_import_v1',reused=True)
        repair=True
        print(f'检测到同源导出缺失 {len(missing)} 个文件，将保留现有数据和标签并补齐。',flush=True)
    if progress:progress(0,len(rows))
    out.parent.mkdir(parents=True,exist_ok=True);stage=Path(tempfile.mkdtemp(prefix=f'.{out.name}.',dir=out.parent))
    try:
        (stage/'image_2').mkdir();(stage/'velodyne').mkdir();decoded=[]
        for local,row in enumerate(rows):
            with np.load(raw/row['path'],allow_pickle=False) as z:a={k:z[k] for k in z.files}
            imgpath=Path(row['image_path']);imgpath=imgpath if imgpath.is_absolute() else raw/imgpath
            imagebytes=imgpath.read_bytes()
            decoded.append((row,a,imagebytes,None))
        if not decoded:raise ValueError('raw import 没有帧')
        fixed=all(np.allclose(a['cam_intrinsic'],decoded[0][1]['cam_intrinsic']) and np.allclose(a['cam_extrinsic'],decoded[0][1]['cam_extrinsic']) for _,a,_,_ in decoded)
        if not fixed:(stage/'calib').mkdir()
        first=int(decoded[0][1]['lidar_timestamp_ns']);prows=[];posevalues=[];fullposes=[]
        with ThreadPoolExecutor(max_workers=4) as images:
            for local,(row,a,jpeg,image) in enumerate(decoded):
                stem=f'{local:06d}'
                if local%4==0:
                    pending=[images.submit(_write_png,decoded[n][2],stage/'image_2'/f'{n:06d}.png') for n in range(local,min(local+4,len(decoded)))]
                pending[local%4].result()
                pts=np.asarray(a['lidar_points']);inten=pts[:,3] if pts.shape[1]>=4 else np.zeros(len(pts),np.float32);np.column_stack((pts[:,:3],inten)).astype('<f4').tofile(stage/'velodyne'/f'{stem}.bin')
                K=a['cam_intrinsic'];E=a['cam_extrinsic'];body='P2: '+' '.join(map(str,np.c_[K,np.zeros(3)].ravel()))+'\nTr: '+' '.join(map(str,E[:3].ravel()))+'\n'
                if fixed:
                    if local==0:(stage/'calib.txt').write_text(body)
                else:(stage/'calib'/f'{stem}.txt').write_text(body)
                pose=np.asarray(a.get('pose_world',np.full(4,np.nan)),float);posevalues.append(pose)
                if np.isfinite(pose).all():
                    c,s=np.cos(pose[3]),np.sin(pose[3]);T=np.eye(4);T[:3,:3]=[[c,-s,0],[s,c,0],[0,0,1]];T[:3,3]=pose[:3];fullposes.append(T)
                prows.append(dict(local_frame_id=local,global_frame_id=int(row['frame_id']),timestamp_ns=int(a['lidar_timestamp_ns']),image_timestamp_ns=int(a.get('image_timestamp_ns',a['lidar_timestamp_ns'])),source_record=row.get('source_index_record',{}),point_columns=int(pts.shape[1])))
                if progress and ((local+1)%25==0 or local+1==len(decoded)):progress(local+1,len(decoded))
        ns=[r['timestamp_ns'] for r in prows];(stage/'times.txt').write_text(''.join(f'{(x-first)/1e9:.9f}\n' for x in ns))
        if len(fullposes)==len(decoded):
            E=np.eye(4);E[:3]=decoded[0][1]['cam_extrinsic'][:3];camera=[E@np.linalg.inv(fullposes[0])@p@np.linalg.inv(E) for p in fullposes];(stage/'poses.txt').write_text(''.join(' '.join(f'{v:.12g}' for v in t[:3].ravel())+'\n' for t in camera))
        pose_path=stage/'poses.txt';poses_hash=hashlib.sha256(pose_path.read_bytes()).hexdigest() if pose_path.is_file() else ''
        times_hash=hashlib.sha256((stage/'times.txt').read_bytes()).hexdigest()
        metadata={'provenance':np.array(json.dumps(dict(format='lidar_raw_import_kitti_v1',source=str(raw),records=prows,status='converted; no labels or ground truth exported'),ensure_ascii=False)),'global_frame_ids':np.array([r['global_frame_id'] for r in prows],np.int64),'local_frame_ids':np.array([r['local_frame_id'] for r in prows],np.int64),'lidar_timestamp_ns':np.array(ns,np.int64),'image_timestamp_ns':np.array([r['image_timestamp_ns'] for r in prows],np.int64),'source_pose_world':np.asarray(posevalues),'poses_txt_sha256':np.array(poses_hash),'times_txt_sha256':np.array(times_hash),'cam_intrinsic':decoded[0][1]['cam_intrinsic'],'cam_extrinsic':decoded[0][1]['cam_extrinsic']}
        for i,(_,a,jpeg,_) in enumerate(decoded):metadata[f'points_{i}']=a['lidar_points'];metadata[f'jpeg_{i}']=np.frombuffer(jpeg,np.uint8)
        np.savez_compressed(stage/'metadata.npz',**metadata)
        manifest={str(p.relative_to(stage)):hashlib.sha256(p.read_bytes()).hexdigest() for p in stage.rglob('*') if p.is_file()}
        if repair:
            missing=_missing_export_files(out,marker)
            if missing-set(manifest):raise ValueError('部分缺失文件无法从当前缓存恢复，已保留现有数据')
            repaired_manifest=dict(marker['output_manifest'])
            additions={name:sha for name,sha in manifest.items() if not (out/name).exists()}
            repaired_manifest.update(additions)
            # Record the new checksums first: interruption leaves valid files or
            # recoverable missing files, never a stale checksum for a repaired file.
            temporary=stage/marker_name
            temporary.write_text(json.dumps(dict(source=origin,source_signature=signature,output_manifest=repaired_manifest),indent=2));temporary.replace(marker_path)
            for name in sorted(additions,key=lambda value:value=='metadata.npz'):
                target=out/name;target.parent.mkdir(parents=True,exist_ok=True);(stage/name).replace(target);repaired_files+=1
            shutil.rmtree(stage)
        else:
            (stage/marker_name).write_text(json.dumps(dict(source=origin,source_signature=signature,output_manifest=manifest),indent=2))
            if out.exists():out.rmdir()
            stage.rename(out)
    except BaseException:shutil.rmtree(stage,ignore_errors=True);raise
    return dict(status='complete',output=str(out),exported_frames=len(rows),frame_ids=[r['frame_id'] for r in rows],input_format='lidar_raw_import_v1',repaired_files=repaired_files)


def _calibration(path,camera='image_2'):
    data={}
    for line in path.read_text().splitlines():
        if ':' in line:
            k,v=line.split(':',1)
            try:data[k.strip()]=np.array([float(x) for x in v.split()])
            except ValueError:pass
    pkey='P'+camera.removeprefix('image_')
    if pkey not in data or data[pkey].size!=12:raise ValueError(f'{path} 缺少相机 {camera} 对应的有效 {pkey}')
    P=data[pkey].reshape(3,4);K=P[:,:3];n=camera.removeprefix('image_')
    common=data.get('Tr',data.get('Tr_velo_to_cam'))
    Tr=data.get(f'Tr_velo_to_cam_{n}',data.get(f'Tr_{n}',common));Tr0=data.get('Tr_velo_to_cam_0',data.get('Tr_0',common))
    if Tr is None or Tr.size!=12 or Tr0 is None or Tr0.size!=12:raise ValueError('calib.txt 缺少相机外参 Tr_velo_to_cam[_n] 或共同 Tr')
    E=np.eye(4);E[:3]=Tr.reshape(3,4);E0=np.eye(4);E0[:3]=Tr0.reshape(3,4);R=data.get('R0_rect',np.eye(3).ravel())
    if R.size!=9:raise ValueError('R0_rect 无效')
    rect=np.eye(4);rect[:3,:3]=R.reshape(3,3)
    rectE=rect@E0;projection=rect@E;projection[:3,3]+=np.linalg.solve(K,P[:,3])
    return K,projection,rectE


def read_kitti(source,output,frames=None,preview=False,dry_run=False,camera=None):
    source,records=frame_records(source,camera);camera=records[0]['camera'];wanted=None if frames is None else set(map(int,frames));chosen=[r for r in records if wanted is None or r['global_frame_id'] in wanted]
    if wanted is not None and wanted-{r['global_frame_id'] for r in records}:raise ValueError('指定帧号不在 KITTI sequence 中')
    output=Path(output).expanduser().absolute()
    if output==source or source in output.parents:raise ValueError('输出目录必须在 KITTI 源目录之外')
    if dry_run:return dict(status='dry_run',format='kitti_odometry',source=str(source),selected_frames=len(chosen),output=str(output),source_readonly=True,camera=camera)
    if output.exists() and any(output.iterdir()):raise ValueError('输出目录必须不存在或为空')
    metadata_path=source/'metadata.npz';meta=None
    if metadata_path.is_file():
        with np.load(metadata_path,allow_pickle=False) as z:meta={k:z[k].copy() for k in z.files}
    times_ns=meta.get('lidar_timestamp_ns') if meta is not None else None
    if not (source/'times.txt').is_file():times_ns=None
    if times_ns is not None and 'times_txt_sha256' in meta:
        tp=source/'times.txt';current=hashlib.sha256(tp.read_bytes()).hexdigest() if tp.is_file() else ''
        if current!=str(meta['times_txt_sha256'].item()):times_ns=None
    pose_txt=[]
    if (source/'poses.txt').is_file():
        for line in (source/'poses.txt').read_text().splitlines():
            vals=np.fromstring(line,sep=' ')
            if vals.size!=12:raise ValueError('poses.txt 每行必须包含 12 个数')
            T=np.eye(4);T[:3]=vals.reshape(3,4);pose_txt.append(T)
        if any(r['local_frame_id']>=len(pose_txt) for r in chosen):raise ValueError('poses.txt 缺少所选 KITTI 帧')
    root_cal=source/'calib.txt'
    if not root_cal.is_file():
        root_cal=source/'calib'/camera/f'{records[0]["local_frame_id"]:06d}.txt'
        if not root_cal.is_file():root_cal=source/'calib'/f'{camera}.txt'
        if not root_cal.is_file():root_cal=source/'calib'/f'{records[0]["local_frame_id"]:06d}.txt'
    _,_,pose_reference=_calibration(root_cal,camera)
    T0=np.eye(4)
    if pose_txt:T0=np.linalg.inv(pose_reference)@pose_txt[0]@pose_reference
    checked_cal=[]
    for r in chosen:
        per_frame=source/'calib'/camera/f'{r["local_frame_id"]:06d}.txt';per_camera=source/'calib'/f'{camera}.txt';legacy_frame=source/'calib'/f'{r["local_frame_id"]:06d}.txt'
        cp=per_frame if per_frame.is_file() else per_camera if per_camera.is_file() else legacy_frame if legacy_frame.is_file() else source/'calib.txt'
        Kp,Ep,Rp=_calibration(cp,camera)
        if not np.isfinite(Kp).all() or abs(np.linalg.det(Kp))<1e-12 or not np.isfinite(Ep).all() or abs(np.linalg.det(Ep[:3,:3]))<1e-9:raise ValueError(f'帧 {r["global_frame_id"]} 标定无效')
        if np.fromfile(r['points_path'],dtype='<f4').size%4:raise ValueError(f'帧 {r["global_frame_id"]} 点云不是 Nx4 little-endian float32')
        checked_cal.append((cp,Kp,Ep,Rp))
    output.mkdir(parents=True,exist_ok=True);(output/'frames').mkdir();(output/'images').mkdir();rows=[];poses=[];transforms=[];timestamps=[];ids=[]
    for ordinal,r in enumerate(chosen):
        i=r['global_frame_id'];local_id=r['local_frame_id'];stem=f'{i:06d}';image_path=r['image_path'];raw_image=image_path.read_bytes()
        with Image.open(io.BytesIO(raw_image)) as im:image=im.convert('RGB')
        raw_pts=np.fromfile(r['points_path'],dtype='<f4')
        if raw_pts.size%4:raise ValueError(f'帧 {i} 点云不是 Nx4 little-endian float32')
        pts=raw_pts.reshape(-1,4);orig=pts.copy()
        mi=r['metadata_index'];candidate=meta.get(f'points_{mi}') if meta is not None else None
        if candidate is not None and candidate.shape[0]==len(pts) and np.array_equal(candidate[:,:3],pts[:,:3]) and ((candidate.shape[1]<4 and np.all(pts[:,3]==0)) or (candidate.shape[1]>=4 and np.array_equal(candidate[:,3],pts[:,3]))):
            orig=candidate.copy()
        calib_path,K,E,_=checked_cal[ordinal]
        camera_xyz=orig[:,:3]@E[:3,:3].T+E[:3,3];h=camera_xyz@K.T;uv=np.full((len(orig),2),np.nan,np.float32);good=camera_xyz[:,2]>.1;uv[good]=h[good,:2]/h[good,2:]
        ns=int(times_ns[mi]) if times_ns is not None else r['timestamp_ns']
        image_ns=int(meta['image_timestamp_ns'][mi]) if meta is not None and 'image_timestamp_ns' in meta else ns
        T=np.full((4,4),np.nan);pose=np.full(4,np.nan)
        if pose_txt:
            T=np.linalg.inv(pose_reference)@pose_txt[local_id]@pose_reference;T=np.linalg.inv(T0)@T
            xyz=T[:3,3];pose=np.array([*xyz,np.arctan2(T[1,0],T[0,0])])
        arrays=dict(xyz=orig[:,:3].astype(np.float32),lidar_points=orig,cam_intrinsic=K,cam_extrinsic=E,pixels_uv=uv,camera_depth=camera_xyz[:,2].astype(np.float32),in_image=good&(uv[:,0]>=0)&(uv[:,0]<image.width)&(uv[:,1]>=0)&(uv[:,1]<image.height),lidar_timestamp_ns=np.asarray(ns if ns is not None else np.nan,dtype=np.int64 if ns is not None else np.float64),image_timestamp_ns=np.asarray(image_ns if image_ns is not None else np.nan,dtype=np.int64 if image_ns is not None else np.float64),pose_world=pose)
        (output/'images'/f'{stem}{image_path.suffix.lower()}').write_bytes(raw_image);np.savez_compressed(output/'frames'/f'{stem}.npz',**arrays)
        if preview:
            from .lmdb_input import preview_frame
            preview_dir=output/'review'/'visualization';preview_dir.mkdir(parents=True,exist_ok=True);preview_frame(image,arrays,preview_dir/f'{stem}.png',i)
        sha=hashlib.sha256()
        for b in (raw_image,r['points_path'].read_bytes(),calib_path.read_bytes(),(root_cal.read_bytes() if root_cal.exists() else b''),(source/'times.txt').read_bytes() if (source/'times.txt').exists() else b'',(source/'poses.txt').read_bytes() if (source/'poses.txt').exists() else b'',metadata_path.read_bytes() if metadata_path.exists() else b''):sha.update(b)
        sha.update(camera.encode())
        record=dict(global_frame_id=i,sequence=str(source),segment=0,camera=camera)
        rows.append(dict(frame_id=i,path=f'frames/{stem}.npz',image_path=str(output/'images'/f'{stem}{image_path.suffix.lower()}'),original_payload_sha256=sha.hexdigest(),source_index_record=record));poses.append(pose);transforms.append(T);timestamps.append(ns);ids.append(i)
    timestamps_array=np.asarray(timestamps,dtype=np.float64) if any(t is None for t in timestamps) else np.asarray(timestamps,np.int64)
    np.savez_compressed(output/'odometry.npz',poses=np.asarray(poses),transforms=np.asarray(transforms),timestamps=timestamps_array,frame_ids=np.asarray(ids,np.int64))
    (output/'index.json').write_text(json.dumps(dict(schema='lidar_raw_import_v1',rows=rows),indent=2))
    if preview:
        review=output/'review';review.mkdir(exist_ok=True);(review/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>KITTI raw input review</title>'+''.join(f'<h2>{i:06d}</h2><img src="visualization/{i:06d}.png">' for i in ids))
    report=dict(status='complete',source=str(source),source_readonly=True,input_format='kitti',camera=camera,imported_frames=len(rows),frame_ids=ids,output=str(output),labels_generated=False,preview=preview,timestamps_available=all(t is not None for t in timestamps))
    (output/'report.json').write_text(json.dumps(report,indent=2));(output/'status.json').write_text(json.dumps(report));return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--count',type=int,default=10);a=p.parse_args(argv)
    print(json.dumps(export_lmdb_kitti(a.source,a.output,a.count),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
