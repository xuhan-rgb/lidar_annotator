"""Full-sequence SAM3 person detection/tracking; map original XY with SAM video identities."""
import argparse,json,shutil,sys,time
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def offload_tracker_state(model):
    original=model.tracker.init_state
    def initialize(*args,**kwargs):
        kwargs["offload_state_to_cpu"]=True
        return original(*args,**kwargs)
    model.tracker.init_state=initialize


def prune_exported_output_cache(state,index,keep=32):
    for f in list(state["cached_frame_outputs"]):
        if f<index-keep:del state["cached_frame_outputs"][f]


def split_temporal_chunks(timestamps_ns, max_gap_ns=2_000_000_000):
    """Split in source order at large gaps or non-increasing timestamps."""
    timestamps=np.asarray(timestamps_ns,dtype=np.int64)
    if not len(timestamps):return []
    chunks=[];start=0
    for index in range(1,len(timestamps)):
        delta=int(timestamps[index])-int(timestamps[index-1])
        if delta<=0 or delta>max_gap_ns:
            chunks.append(list(range(start,index)));start=index
    chunks.append(list(range(start,len(timestamps))))
    return chunks


def remap_object_ids(native_ids, offset):
    """Give native IDs from this independently initialized chunk global IDs."""
    ids=np.asarray(native_ids,dtype=np.int64)
    return ids+int(offset)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);a=p.parse_args();c=json.loads(a.config.read_text());root=Path(c['temporal_output']);person=Path(c['person_root']);out=root/'map_review/sam3_full_video';(out/'images').mkdir(parents=True,exist_ok=True);(out/'masks').mkdir(exist_ok=True);(out/'associations').mkdir(exist_ok=True)
    here=Path(__file__).resolve().parents[1];sys.path.insert(0,str(here/'semantic_zgc'))
    from common import read_source_image
    from sam3.model_builder import build_sam3_video_model
    recipe=json.loads((here/'zgc_mask_recipe_v1/recipe.json').read_text());rows=json.loads(Path(c['source_index']).read_text())['rows']
    source_root=Path(c['source_index']).parent
    timestamps=[]
    for row in rows:
        with np.load(source_root/row['path'],allow_pickle=False) as z:timestamps.append(int(np.asarray(z['lidar_timestamp_ns']).item()))
    chunks=split_temporal_chunks(timestamps)
    video_inputs=out/'images';video_inputs.mkdir(parents=True,exist_ok=True)
    for stale_image in video_inputs.glob('*.jpg'):stale_image.unlink()
    image_shapes={}
    for chunk_id,indices in enumerate(chunks):
        image_dir=video_inputs/f'chunk_{chunk_id:05d}'
        if image_dir.exists():shutil.rmtree(image_dir)
        image_dir.mkdir(parents=True)
        for local_index,global_index in enumerate(indices):
            image=read_source_image(rows[global_index]);image_shapes[global_index]=(image.height,image.width)
            image.save(image_dir/f'{local_index:05d}.jpg',quality=95)
        print(f'video chunk {chunk_id+1}/{len(chunks)} ready frames={len(indices)}',flush=True)
    from runtime_paths import MODEL_CACHE
    checkpoint=str(MODEL_CACHE/'sam3.pt');started=time.time();model=build_sam3_video_model(checkpoint_path=checkpoint,load_from_HF=False,strict_state_dict_loading=True,compile=False).cuda().eval();offload_tracker_state(model);print('full video model loaded; tracker history offloaded to CPU',flush=True)
    processed=set();matched_total=0;raw_total=0;object_ids=set();id_offset=0
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        def cpu(value):return value.detach().cpu().numpy() if isinstance(value,torch.Tensor) else np.asarray(value)
        for chunk_id,indices in enumerate(chunks):
            state=model.init_state(str(video_inputs/f'chunk_{chunk_id:05d}'),offload_video_to_cpu=True)
            _,initial=model.add_prompt(state,frame_idx=0,text_str='person')
            chunk_max_native_id=-1
            for local_index,outputs in model.propagate_in_video(state,start_frame_idx=0,max_frame_num_to_track=len(indices)-1,reverse=False):
                global_index=indices[int(local_index)];f=int(Path(rows[global_index]['path']).stem)
                native_ids=cpu(outputs['out_obj_ids']).astype(int);masks=cpu(outputs['out_binary_masks']).astype(bool)
                if masks.ndim==4:masks=masks[:,0]
                if len(native_ids):
                    shape=masks.shape[-2:];ids=remap_object_ids(native_ids,id_offset)
                    chunk_max_native_id=max(chunk_max_native_id,int(native_ids.max()))
                else:
                    shape=image_shapes[global_index];masks=np.empty((0,*shape),bool);ids=native_ids
                object_ids.update(ids.tolist());processed.add(f)
                np.savez_compressed(out/'masks'/f'{f:06d}.npz',object_ids=ids,mask_shape=shape,masks_packed=np.packbits(masks.reshape(len(ids),int(np.prod(shape))),axis=1,bitorder='little'))
                with np.load(person/'frames'/f'{f:06d}.npz') as z:raw_iids=z['xy_instance_ids'];raw_xy=z['xy']
                with np.load(person/'raw_sam_person/frames'/f'{f:06d}.npz') as z:s=tuple(z['mask_shape']);old=np.unpackbits(z['masks_packed'],axis=1,count=int(np.prod(s)),bitorder='little').reshape(-1,*s).astype(bool)
                overlap=np.zeros((len(raw_iids),len(ids)))
                for i,iid in enumerate(raw_iids):
                    for j,m in enumerate(masks):
                        union=np.count_nonzero(old[iid]|m);overlap[i,j]=np.count_nonzero(old[iid]&m)/union if union else 0.
                assignments={}
                if overlap.size:
                    ii,jj=linear_sum_assignment(-overlap)
                    assignments={int(i):int(j) for i,j in zip(ii,jj) if overlap[i,j]>=.3}
                associations=[]
                for i,(iid,xy) in enumerate(zip(raw_iids,raw_xy)):
                    j=assignments.get(i);associations.append(dict(instance_id=int(iid),raw_xy=xy.tolist(),sam_video_id=int(ids[j]) if j is not None else None,mask_iou=float(overlap[i,j]) if j is not None else None))
                matched_total+=len(assignments);raw_total+=len(raw_iids);(out/'associations'/f'{f:06d}.json').write_text(json.dumps(dict(frame=f,observations=associations),indent=2))
                prune_exported_output_cache(state,local_index)
                if len(processed)%50==0:print(f'tracked {len(processed)}/{len(rows)}; matched original XY {matched_total}/{raw_total}; elapsed {time.time()-started:.1f}s',flush=True)
            if chunk_max_native_id>=0:id_offset+=chunk_max_native_id+1
            print(f'tracked chunk {chunk_id+1}/{len(chunks)}; frames={len(indices)}; global_id_offset={id_offset}',flush=True)
    assert len(processed)==len(rows)
    report=dict(status='complete',frame_count=len(processed),chunk_count=len(chunks),max_temporal_gap_seconds=2,sam_video_object_ids=len(object_ids),raw_positions=raw_total,matched_raw_positions=matched_total,unmatched_raw_positions=raw_total-matched_total,checkpoint=checkpoint,model='official SAM3 video detector + memory tracker, initialized per contiguous chunk',prompt='person',mask_match_iou=.3,identity_source='SAM3 video output object IDs, offset per independently initialized chunk; Hungarian matching only maps each frame original instances to output masks',initialization='SAM3 text discovery and native new-object lifecycle within contiguous source-time chunks; existing YOLO-confirmed raw XY retained for display',prediction=False,smoothing=False,labels_modified=False,elapsed_seconds=round(time.time()-started,2),limits='Map positions remain original per-frame XY. Unmatched observations stay gray. No appearance identity ground truth. New SAM detections without saved raw XY are not added to map.')
    (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
