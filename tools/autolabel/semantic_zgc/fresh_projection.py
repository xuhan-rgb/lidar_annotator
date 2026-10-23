import numpy as np

def project_labels(arrays,mask,confidence):
    """Rebuild visible image supervision; never inherit old point-label ignores."""
    uv=arrays['pixels_uv'];h,w=mask.shape
    finite=np.isfinite(uv).all(1);pixels=np.zeros(uv.shape,dtype=np.int64);pixels[finite]=np.floor(uv[finite]).astype(np.int64)
    valid=np.asarray(arrays['visible'],bool)&finite&(pixels[:,0]>=0)&(pixels[:,0]<w)&(pixels[:,1]>=0)&(pixels[:,1]<h)
    ids=np.flatnonzero(valid);q=pixels[ids]
    labels=np.full(len(uv),255,np.uint8);scores=np.zeros(len(uv),np.float32)
    labels[ids]=mask[q[:,1],q[:,0]];scores[ids]=confidence[q[:,1],q[:,0]];scores[labels==255]=0
    return labels,scores
