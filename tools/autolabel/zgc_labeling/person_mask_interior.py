"""Linear-time mask-interior half selection for observed pedestrian LiDAR points."""
import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes


def select_interior_half(mask,pixels_uv,source_ids):
    """Keep ceil(N/2) valid points farthest from the outer boundary, inside the original mask."""
    mask=np.asarray(mask,bool);ids=np.asarray(source_ids,int);uv=np.asarray(pixels_uv)[ids]
    finite=np.isfinite(uv).all(1);pixels=np.floor(np.where(np.isfinite(uv),uv,0)).astype(int)
    h,w=mask.shape;valid=finite&(pixels[:,0]>=0)&(pixels[:,0]<w)&(pixels[:,1]>=0)&(pixels[:,1]<h)
    ix=np.flatnonzero(valid);valid[ix]&=mask[pixels[ix,1],pixels[ix,0]];ids=ids[valid];pixels=pixels[valid]
    if not len(ids):return ids,np.empty(0),dict(valid_points=0,selected_points=0,crop_pixels=0)
    x,y,width,height=cv2.boundingRect(mask.view(np.uint8))
    crop=np.pad(binary_fill_holes(mask[y:y+height,x:x+width]).astype(np.uint8),1)
    # 5x5 chamfer approximation to Euclidean boundary distance; no pixel-pair distances.
    distance=cv2.distanceTransform(crop,cv2.DIST_L2,cv2.DIST_MASK_5)
    scores=distance[pixels[:,1]-y+1,pixels[:,0]-x+1];keep=(len(ids)+1)//2
    selected=np.argpartition(-scores,keep-1)[:keep]
    return ids[selected],scores,dict(valid_points=len(ids),selected_points=keep,crop_pixels=crop.size,mask_bbox_xywh=[x,y,width,height],distance_method='OpenCV L2 5x5 chamfer, filled outer-boundary ROI',selection='top ceil(N/2) boundary distances via argpartition')
