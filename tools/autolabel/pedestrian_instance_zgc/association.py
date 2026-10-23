"""SAM-person instance deduplication, one-to-one 2D matching, and exclusive LiDAR support."""
import numpy as np
from scipy.optimize import linear_sum_assignment

ASSOC_BOX_IOU_MIN=.1
ASSOC_MASK_COVERAGE_MIN=.25
ASSOC_UTILITY_MARGIN_MIN=.05
MASK_DEDUP_IOU=.9
MASK_DEDUP_BOX_IOU=.9


def box_intersection(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    return float(max(0.,min(a[2],b[2])-max(a[0],b[0]))*max(0.,min(a[3],b[3])-max(a[1],b[1])))


def box_iou(a,b):
    inter=box_intersection(a,b)
    a=np.asarray(a,float);b=np.asarray(b,float)
    aa=max(0.,a[2]-a[0])*max(0.,a[3]-a[1]);bb=max(0.,b[2]-b[0])*max(0.,b[3]-b[1])
    return inter/max(aa+bb-inter,1e-12)


def mask_bbox(mask):
    y,x=np.nonzero(np.asarray(mask,bool))
    return [float(x.min()),float(y.min()),float(x.max()+1),float(y.max()+1)] if len(x) else [0.,0.,0.,0.]


def mask_iou(a,b):
    a=np.asarray(a,bool);b=np.asarray(b,bool)
    if a.shape!=b.shape:raise ValueError('SAM masks must have the same image shape')
    inter=int(np.count_nonzero(a&b));union=int(np.count_nonzero(a|b))
    return inter/max(union,1)


def deduplicate_masks(masks,records,mask_iou_min=MASK_DEDUP_IOU,bbox_iou_min=MASK_DEDUP_BOX_IOU):
    """Drop only near-identical same-prompt masks; never union distinct person instances."""
    keep=[];dropped=[]
    order=sorted(range(len(masks)),key=lambda i:float(records[i]['score']),reverse=True)
    for i in order:
        duplicate=next((j for j in keep if mask_iou(masks[i],masks[j])>=mask_iou_min
                        and box_iou(records[i]['bbox_xyxy'],records[j]['bbox_xyxy'])>=bbox_iou_min),None)
        if duplicate is None:keep.append(i)
        else:dropped.append(dict(source_instance_id=int(records[i]['sam_instance_id']),
            duplicate_of_source_instance_id=int(records[duplicate]['sam_instance_id']),reason='near_identical_same_prompt_mask'))
    keep=sorted(keep)
    out_masks=[];out_records=[]
    for index,src in enumerate(keep):
        out_masks.append(np.asarray(masks[src],bool))
        rec=dict(records[src]);rec['source_instance_id']=int(rec['sam_instance_id']);rec['instance_id']=index
        rec['duplicate_source_ids']=[x['source_instance_id'] for x in dropped if x['duplicate_of_source_instance_id']==rec['sam_instance_id']]
        out_records.append(rec)
    return out_masks,out_records,dropped


def associate_boxes_to_masks(box_detections,masks,instances,
                              min_box_iou=ASSOC_BOX_IOU_MIN,
                              min_mask_coverage=ASSOC_MASK_COVERAGE_MIN,
                              utility_margin=ASSOC_UTILITY_MARGIN_MIN):
    """Hungarian one-to-one association of detections and SAM instances.

    Mask coverage is the fraction of the full instance raster lying inside the detector box.
    A small margin over the next eligible match rejects ambiguous correspondences.
    """
    nd,nm=len(box_detections),len(masks)
    metrics=np.zeros((nd,nm,4),dtype=float);eligible=np.zeros((nd,nm),bool)
    for di,d in enumerate(box_detections):
        dbox=np.asarray(d['bbox_xyxy'],float);area=max((dbox[2]-dbox[0])*(dbox[3]-dbox[1]),1e-12)
        for mi,mask in enumerate(masks):
            ibox=np.asarray(instances[mi]['bbox_xyxy'],float);inter=box_intersection(dbox,ibox)
            mask_area=max(int(np.asarray(mask,bool).sum()),1)
            x0,y0=max(0,int(np.floor(dbox[0]))),max(0,int(np.floor(dbox[1])))
            x1,y1=min(mask.shape[1],int(np.ceil(dbox[2]))),min(mask.shape[0],int(np.ceil(dbox[3])))
            raster=float(np.asarray(mask,bool)[y0:y1,x0:x1].sum()/mask_area) if x1>x0 and y1>y0 else 0.
            iou=box_iou(dbox,ibox)
            detcov=inter/area
            maskbboxcov=inter/max((ibox[2]-ibox[0])*(ibox[3]-ibox[1]),1e-12)
            metrics[di,mi]=[iou,maskbboxcov,detcov,raster]
            eligible[di,mi]=(iou>=min_box_iou and raster>=min_mask_coverage)
    utility=(metrics[:,:,0]+metrics[:,:,3])/2
    if nd and nm:
        rows,cols=linear_sum_assignment(-np.where(eligible,utility,-1e6))
        proposed={int(r):int(c) for r,c in zip(rows,cols) if eligible[r,c]}
    else:proposed={}
    ambiguous_rows=set();ambiguous_cols=set()
    for di,mi in proposed.items():
        row_alt=utility[di,eligible[di]&(np.arange(nm)!=mi)]
        col_alt=utility[eligible[:,mi]&(np.arange(nd)!=di),mi]
        if ((len(row_alt) and utility[di,mi]-row_alt.max()<utility_margin)
                or (len(col_alt) and utility[di,mi]-col_alt.max()<utility_margin)):
            ambiguous_rows.add(di);ambiguous_cols.add(mi)
    assigned={d:m for d,m in proposed.items() if d not in ambiguous_rows and m not in ambiguous_cols}
    rows_out=[]
    for di,d in enumerate(box_detections):
        if di in ambiguous_rows:status='ignore_ambiguous_instance_match'
        elif di in assigned:status='matched'
        elif any(eligible[di]):status='unmatched_one_to_one_conflict'
        else:status='unmatched_no_eligible_instance'
        rows_out.append(dict(detection_index=int(d.get('id',d.get('index',di))),bbox_xyxy=list(map(float,d['bbox_xyxy'])),
            status=status,instance_id=assigned.get(di),candidates=[dict(instance_id=int(mi),box_iou=float(metrics[di,mi,0]),
                mask_bbox_coverage=float(metrics[di,mi,1]),detection_bbox_coverage=float(metrics[di,mi,2]),
                mask_pixel_coverage=float(metrics[di,mi,3]),utility=float(utility[di,mi]),eligible=bool(eligible[di,mi]))
                for mi in range(nm)]))
    return rows_out,metrics,eligible


def exclusive_masks(masks):
    if len(masks)==0:return [],np.zeros((0,0),np.uint16)
    stack=np.asarray(masks,bool)
    cover=stack.sum(axis=0,dtype=np.uint16)
    return [m&(cover==1) for m in stack],cover


def support_point_indices(xyz,pixels_uv,visible,camera_depth,masks,roi=(-20,-20,-3,20,20,6)):
    """Assign each valid LiDAR return to at most one exclusive SAM instance."""
    xyz=np.asarray(xyz,float);uv=np.asarray(pixels_uv,float);visible=np.asarray(visible,bool);depth=np.asarray(camera_depth,float)
    if xyz.ndim!=2 or xyz.shape[1]!=3 or uv.shape!=(len(xyz),2) or len(depth)!=len(xyz) or len(visible)!=len(xyz):
        raise ValueError('point/pixel/depth array shapes do not align')
    masks_unique,cover=exclusive_masks(masks)
    valid=visible&np.isfinite(xyz).all(1)&np.isfinite(uv).all(1)&np.isfinite(depth)&(depth>.1)
    lo=np.asarray(roi[:3]);hi=np.asarray(roi[3:])
    valid&=((xyz>=lo)&(xyz<hi)).all(1)&(np.linalg.norm(xyz,axis=1)>=.5)
    pix=np.floor(np.where(np.isfinite(uv),uv,0.)).astype(np.int64)
    height,width=cover.shape
    valid&=(pix[:,0]>=0)&(pix[:,0]<width)&(pix[:,1]>=0)&(pix[:,1]<height)
    ids=np.flatnonzero(valid);assign=np.full(len(xyz),-1,np.int32)
    for iid,mask in enumerate(masks_unique):
        sel=ids[mask[pix[ids,1],pix[ids,0]]]
        if len(sel):assign[sel]=iid
    return [np.flatnonzero(assign==i) for i in range(len(masks))],assign,cover,valid
