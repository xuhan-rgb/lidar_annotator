"""Shared label/display threshold and exact geometry of saved 7D LiDAR boxes."""
import numpy as np
DISPLAY_SCORE_THRESHOLD = 0.6
DISPLAY_SCORE_3D_THRESHOLD = 0.1

def passes_display_threshold(score, score_3d):
    return float(score) >= DISPLAY_SCORE_THRESHOLD and float(score_3d) >= DISPLAY_SCORE_3D_THRESHOLD

def boxes_to_corners_lidar(boxes):
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 7)
    signs = np.array([[-1,-1,-1],[1,-1,-1],[1,-1,1],[-1,-1,1],[-1,1,-1],[1,1,-1],[1,1,1],[-1,1,1]], dtype=np.float64)
    local = signs[None] * boxes[:,None,3:6] / 2
    cosine, sine = np.cos(boxes[:,6,None]), np.sin(boxes[:,6,None])
    result = local.copy()
    result[:,:,0] = cosine * local[:,:,0] - sine * local[:,:,1]
    result[:,:,1] = sine * local[:,:,0] + cosine * local[:,:,1]
    return result + boxes[:,None,:3]
