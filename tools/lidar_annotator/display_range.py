"""Viewer-only Cartesian bounds; never alter training or source point arrays."""
import numpy as np

DEFAULT_LIMITS=(20.,-3.,6.)


def point_mask(xyz,limits=DEFAULT_LIMITS):
    extent,zmin,zmax=limits
    if not np.isfinite(limits).all() or extent<=0 or zmin>=zmax:
        raise ValueError('XY 范围必须大于 0，高度下限必须小于上限')
    return np.isfinite(xyz).all(1)&(np.abs(xyz[:,:2])<=extent).all(1)&(xyz[:,2]>=zmin)&(xyz[:,2]<=zmax)
