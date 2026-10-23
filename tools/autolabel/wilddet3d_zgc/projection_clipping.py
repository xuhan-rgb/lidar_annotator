"""Safe camera-space edge projection for saved 3D cuboids."""
import numpy as np


def _clip_segment_to_viewport(a, b, width, height):
    """Liang-Barsky clip of a projected segment to pixel-center bounds."""
    x0, y0 = map(float, a)
    x1, y1 = map(float, b)
    dx, dy = x1 - x0, y1 - y0
    t_enter, t_leave = 0.0, 1.0
    for p, q in ((-dx, x0), (dx, width - 1.0 - x0),
                 (-dy, y0), (dy, height - 1.0 - y0)):
        if p == 0.0:
            if q < 0.0:
                return None
            continue
        ratio = q / p
        if p < 0.0:
            t_enter = max(t_enter, ratio)
        else:
            t_leave = min(t_leave, ratio)
        if t_enter > t_leave:
            return None
    clipped_a = (x0 + t_enter * dx, y0 + t_enter * dy)
    clipped_b = (x0 + t_leave * dx, y0 + t_leave * dy)
    clipped_a = (float(np.clip(clipped_a[0], 0., width - 1.)),
                 float(np.clip(clipped_a[1], 0., height - 1.)))
    clipped_b = (float(np.clip(clipped_b[0], 0., width - 1.)),
                 float(np.clip(clipped_b[1], 0., height - 1.)))
    return clipped_a, clipped_b


def clip_camera_edge_to_image(point_a_camera, point_b_camera, intrinsic,
                              width, height, near=0.1):
    """Clip a 3D camera-space edge at z>=near, project, then clip to viewport.

    Returns pixel-endpoint pairs bounded to the image, or None when the edge is
    entirely behind the camera or its projected segment misses the image.
    """
    if width <= 0 or height <= 0 or near <= 0:
        raise ValueError("Image dimensions and near plane must be positive")
    a = np.asarray(point_a_camera, dtype=np.float64).reshape(3).copy()
    b = np.asarray(point_b_camera, dtype=np.float64).reshape(3).copy()
    K = np.asarray(intrinsic, dtype=np.float64).reshape(3, 3)
    if not np.isfinite(a).all() or not np.isfinite(b).all() or not np.isfinite(K).all():
        return None
    za, zb = float(a[2]), float(b[2])
    if za < near and zb < near:
        return None
    if za < near:
        t = (near - za) / (zb - za)
        a += t * (b - a)
        a[2] = near
    elif zb < near:
        t = (near - zb) / (za - zb)
        b += t * (a - b)
        b[2] = near
    uv_a_h = K @ a
    uv_b_h = K @ b
    if (not np.isfinite(uv_a_h).all() or not np.isfinite(uv_b_h).all()
            or uv_a_h[2] <= 0 or uv_b_h[2] <= 0):
        return None
    uv_a = uv_a_h[:2] / uv_a_h[2]
    uv_b = uv_b_h[:2] / uv_b_h[2]
    if not np.isfinite(uv_a).all() or not np.isfinite(uv_b).all():
        return None
    return _clip_segment_to_viewport(uv_a, uv_b, width, height)


def project_clipped_box_edges(corners_camera, edges, intrinsic, width, height, near=0.1):
    """Return visible clipped image edges as (edge-index, xy0, xy1) tuples."""
    corners = np.asarray(corners_camera, dtype=np.float64).reshape(-1, 3)
    result = []
    for edge_index, (a, b) in enumerate(edges):
        clipped = clip_camera_edge_to_image(corners[a], corners[b], intrinsic,
                                            width, height, near=near)
        if clipped is not None:
            result.append((edge_index, clipped[0], clipped[1]))
    return result
