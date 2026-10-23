"""Generate the approved camera-ray + distance-gated vehicle labels at score .3."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import hashlib
from functools import lru_cache
from pathlib import Path
import time

import numpy as np

from instance_label_config import DATA, SOURCE

OUTPUT_ROOT = DATA / "wilddet3d_vehicle_camera_ray_distance030_20261014"
SCORE_THRESHOLD = .3
SCORE_3D_THRESHOLD = .1


@lru_cache(maxsize=1)
def _algorithm_signature():
    """One computation per invocation; radius/solver changes invalidate resume."""
    here=Path(__file__).resolve().parent
    files=('process_vehicle_camera_ray_distance.py','center_distance_selection.py',
           'review_bev_pose.py','associate_instance_masks.py','verify_vehicle_class_000402.py',
           'refine_instance_support.py','instance_label_config.py','label_geometry.py')
    policy=dict(score_threshold=SCORE_THRESHOLD,score_3d_threshold=SCORE_3D_THRESHOLD,
                code_sha256={name:hashlib.sha256((here/name).read_bytes()).hexdigest() for name in files})
    return hashlib.sha256(json.dumps(policy,sort_keys=True).encode()).hexdigest()


def _resume_valid(root, frame, source_hash, expected_signature=None):
    signature=expected_signature or _algorithm_signature()
    signature_path=root/'reports/generation_signature.json'
    if not signature_path.is_file():return False
    if json.loads(signature_path.read_text()).get('signature')!=signature:return False
    name=f"{frame:06d}"
    jp=root/"refined"/f"{name}.json";npz=root/"refined"/f"{name}.npz"
    support=root/"refined/support"/f"{name}.npz"
    if not (jp.is_file() and npz.is_file() and support.is_file()):return False
    meta=json.loads(jp.read_text())
    if (meta.get("source_image_sha256")!=source_hash or meta.get("score_threshold")!=SCORE_THRESHOLD
            or meta.get("score_3d_threshold")!=SCORE_3D_THRESHOLD or meta.get("config_approved") is not True
            or meta.get("generation_signature")!=signature):
        return False
    try:
        with np.load(npz) as a,np.load(support) as s:
            n=int(meta["accepted_count"])
            return (a["box_3d"].shape==(n,7) and a["corners_3d_lidar"].shape==(n,8,3)
                    and len(a["seed_ids"])==n and s["offsets"].shape==(n+1,))
    except (OSError,KeyError,ValueError):return False


def _worker(frame, output_root, source_row):
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)
    from process_vehicle_camera_ray_distance import process_frame
    result=process_frame(frame,output_root,source_row)
    metadata_path=Path(output_root)/'refined'/f'{frame:06d}.json'
    metadata=json.loads(metadata_path.read_text())
    metadata['generation_signature']=_algorithm_signature()
    temp=metadata_path.with_suffix('.signature.tmp.json')
    temp.write_text(json.dumps(metadata,indent=2));temp.replace(metadata_path)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--workers",type=int,default=8)
    parser.add_argument("--no-resume",action="store_true")
    parser.add_argument("--frames",nargs="*",type=int)
    args=parser.parse_args()
    root=OUTPUT_ROOT.resolve()
    for folder in ("refined/support","association","reports"):(root/folder).mkdir(parents=True,exist_ok=True)
    rows=json.loads((SOURCE/"index.json").read_text())["rows"]
    by_frame={int(Path(r["path"]).stem):r for r in rows}
    all_ids=sorted(by_frame)
    frame_ids=all_ids if args.frames is None else sorted(set(args.frames))
    unknown=sorted(set(frame_ids)-set(by_frame))
    if unknown:raise KeyError(f"Unknown source frame IDs: {unknown}")
    pending=[f for f in frame_ids if args.no_resume or not _resume_valid(root,f,by_frame[f]["original_payload_sha256"])]
    signature=dict(signature=_algorithm_signature(),score_threshold=SCORE_THRESHOLD,
                   score_3d_threshold=SCORE_3D_THRESHOLD,
                   purpose='resume only when the frozen solver/selection configuration matches')
    (root/'reports/generation_signature.json').write_text(json.dumps(signature,indent=2))
    progress={"status":"running","total":len(frame_ids),"completed":len(frame_ids)-len(pending),
              "pending":len(pending),"workers":args.workers,"score_threshold":SCORE_THRESHOLD,
              "score_3d_threshold":SCORE_3D_THRESHOLD,
              "completed_frame_ids":[f"{f:06d}" for f in frame_ids if f not in pending],
              "pending_frame_ids":[f"{f:06d}" for f in pending]}
    progress_path=root/"reports/run_progress.json";progress_path.write_text(json.dumps(progress,indent=2))
    started=time.time();results=[]
    with ProcessPoolExecutor(max_workers=max(1,args.workers)) as pool:
        futures={pool.submit(_worker,f,str(root),by_frame[f]):f for f in pending}
        for future in as_completed(futures):
            frame=futures[future];result=future.result();results.append(result)
            progress["completed"]+=1;progress["pending"]-=1;progress["last_completed_frame"]=f"{frame:06d}"
            progress["elapsed_seconds"]=round(time.time()-started,3)
            progress["completed_frame_ids"].append(f"{frame:06d}");progress["pending_frame_ids"].remove(f"{frame:06d}")
            temp=progress_path.with_suffix(".tmp.json");temp.write_text(json.dumps(progress,indent=2));temp.replace(progress_path)
            print(json.dumps({"completed":progress["completed"],"total":len(frame_ids),"frame":f"{frame:06d}",
                              "accepted":result["accepted_count"],"statuses":result["statuses"]}),flush=True)
    totals={};accepted_total=0;nonempty=0;summary_rows=[]
    for frame in frame_ids:
        name=f"{frame:06d}";meta=json.loads((root/"refined"/f"{name}.json").read_text())
        accepted=int(meta["accepted_count"]);accepted_total+=accepted;nonempty+=accepted>0
        statuses=meta["status_counts"]
        for key,value in statuses.items():totals[key]=totals.get(key,0)+int(value)
        summary_rows.append({"frame":name,"accepted_count":accepted,
                             "seed_ids":np.load(root/"refined"/f"{name}.npz")["seed_ids"].astype(int).tolist(),
                             "status_counts":statuses})
    summary={"status":"complete","frame_count":len(frame_ids),"accepted_label_count":accepted_total,
             "frames_with_labels":nonempty,"status_totals":totals,"score_threshold":SCORE_THRESHOLD,
             "score_3d_threshold":SCORE_3D_THRESHOLD,"manual_gt":False,"added_to_training":False,
             "frames":summary_rows}
    (root/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    progress.update(status="complete",pending=0,elapsed_seconds=round(time.time()-started,3))
    progress_path.write_text(json.dumps(progress,indent=2))
    (root/"reports/run_report.json").write_text(json.dumps({"status":"complete","frame_count":len(frame_ids),
        "score_threshold":SCORE_THRESHOLD,"score_3d_threshold":SCORE_3D_THRESHOLD,"workers":args.workers,
        "resumed_frames":len(frame_ids)-len(pending),"elapsed_seconds":round(time.time()-started,3),
        "source_index":str(SOURCE/"index.json"),"source_images":"LMDB original RGB accessed via source rows",
        "raw_predictions":"cached WildDet3D; no re-inference","sam_instances":"cached official SAM3 per-instance masks",
        "yolo":"cached YOLO car/bus/truck category metadata","algorithm":"process_vehicle_camera_ray_distance.py + center_distance_selection.py"},
        indent=2,ensure_ascii=False))
    print(json.dumps({"output_root":str(root),"frame_count":len(frame_ids),"accepted_label_count":accepted_total,
                      "frames_with_labels":nonempty,"status_totals":totals,
                      "elapsed_seconds":round(time.time()-started,3)},indent=2),flush=True)


if __name__=="__main__":main()
