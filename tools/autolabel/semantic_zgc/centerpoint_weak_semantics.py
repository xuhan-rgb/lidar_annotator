"""Confidence-weighted real-data semantics, without pseudo detection targets."""
import json
from pathlib import Path

import numpy as np


# Teacher: ground, person, vehicle, wall/building, vegetation, fence.
# Ground is explicitly supervised as occupancy class 0 (unoccupied).
OCCUPANCY_BITS = np.array([1, 2, 4, 32, 24, 64], dtype=np.uint8)
DRIVABLE_BITS = np.array([2, 1, 1, 1, 0, 1], dtype=np.uint8)


def static_drivable_targets(archive):
    """Separate observed ground affordance from current obstacle occupancy."""
    xyz = np.asarray(archive['xyz'])
    labels = np.asarray(archive['point_labels'])
    confidence = np.clip(archive['point_confidence'], 0, 1)
    valid = np.isfinite(xyz).all(1) & np.isfinite(confidence) & (confidence > 0)
    valid &= ((xyz >= [-40,-40,-3]) & (xyz < [40,40,12])).all(1)
    valid &= (np.linalg.norm(xyz,axis=1) >= .5) & (labels < 7)
    cells = np.floor((xyz[valid,:2]+40)/.25).astype(np.int64)
    scores = np.full((7,320,320),np.inf,np.float32)
    np.minimum.at(scores,(labels[valid],cells[:,0],cells[:,1]),confidence[valid])
    support = np.isfinite(scores)
    occupancy = np.full((320,320),255,np.uint8)
    # Occupancy is never cleared by an observed ground return in the same cell.
    for label in (0,4,3,5,6,2,1):
        occupancy[support[label]] = label
    bits = np.zeros((320,320),np.uint8)
    occ_conf = np.zeros((320,320),np.float32)
    for label,bit in enumerate((1,2,4,32,16,64,8)):
        selected = occupancy == label
        bits[selected] = bit
        occ_conf[selected] = scores[label,selected]
    drive = np.zeros((320,320),np.uint8)
    drive_conf = np.zeros((320,320),np.float32)
    drive[support[0]] = 2
    drive_conf[support[0]] = scores[0,support[0]]
    # Actors do not imply hidden drivable ground. Static obstacles block it.
    for label in (3,5,6):
        drive[support[label]] = 1
        drive_conf[support[label]] = scores[label,support[label]]
    return {name:np.ascontiguousarray(value.T) for name,value in (
        ('weak_occupancy_bits',bits),('weak_confidence',occ_conf),
        ('weak_drivable_bits',drive),('weak_drivable_confidence',drive_conf))}


def weak_targets(archive, schema="zgc_image_teacher_six_v1"):
    """Convert ascending [X,Y] teacher grids to network [Y,X], without reversal."""
    seven = schema == 'zgc_image_teacher_seven_v2'
    if schema not in ('zgc_image_teacher_six_v1', 'zgc_image_teacher_seven_v2'):
        raise ValueError(f'Unsupported weak label schema: {schema}')
    if str(archive.get('supervision_policy','')) == 'static_drivable_occupancy_v1':
        if not seven:
            raise ValueError('Static drivable policy requires seven-class teacher labels')
        return static_drivable_targets(archive)
    class_count = 7 if seven else 6
    occupancy_mapping = np.array([1, 2, 4, 32, 16, 64, 8], dtype=np.uint8) if seven else OCCUPANCY_BITS
    drivable_mapping = np.array([2, 1, 1, 1, 0, 1, 1], dtype=np.uint8) if seven else DRIVABLE_BITS
    labels = archive['bev_labels']
    if labels.shape != (320, 320):
        raise ValueError('Weak BEV labels must be 320x320')
    xyz = archive['xyz']
    point_labels = archive['point_labels']
    confidence = np.clip(archive['point_confidence'], 0, 1)
    keep = np.isfinite(xyz).all(1) & np.isfinite(confidence)
    keep &= ((xyz >= [-40, -40, -3]) & (xyz < [40, 40, 12])).all(1)
    keep &= np.linalg.norm(xyz, axis=1) >= .5
    keep &= point_labels < class_count
    cells = np.floor((xyz[keep, :2] + 40) / .25).astype(np.int64)
    kept_labels = point_labels[keep]
    # Resolve all heights using ground > person > vehicle > fence > wall > tree.
    support = np.zeros((class_count, *labels.shape), dtype=bool)
    np.logical_or.at(support, (kept_labels, cells[:, 0], cells[:, 1]), True)
    resolved = labels.copy()
    for label in ((4, 6, 3, 5, 2, 1, 0) if seven else (4, 3, 5, 2, 1, 0)):
        resolved[support[label]] = label
    # Binary supervision retains its previous labels and confidence independently.
    drive_labels = archive['bev_drivable_teacher_labels'] if 'bev_drivable_teacher_labels' in archive else labels
    drive_labels = drive_labels.copy()
    has_actor = support[1:3].any(0) | np.isin(drive_labels, (1, 2))
    drive_labels[support[3] & support[5] & ~has_actor] = 255
    drive_labels[support[3] & support[4] & ~support[5] & ~has_actor] = 3
    drive_labels[support[5] & support[4] & ~support[3] & ~has_actor] = 5

    def targets_and_confidence(selected, mapping):
        valid = selected < class_count
        targets = np.zeros_like(labels)
        targets[valid] = mapping[selected[valid]]
        matching = selected[cells[:, 0], cells[:, 1]] == kept_labels
        cell_confidence = np.full(labels.shape, np.inf, dtype=np.float32)
        matching_cells = cells[matching]
        np.minimum.at(cell_confidence, (matching_cells[:, 0], matching_cells[:, 1]), confidence[keep][matching])
        supported = valid & np.isfinite(cell_confidence)
        cell_confidence[~supported] = 0
        targets[~supported] = 0
        return targets, cell_confidence

    occupancy, cell_confidence = targets_and_confidence(resolved, occupancy_mapping)
    drivable, drive_confidence = targets_and_confidence(drive_labels, drivable_mapping)
    result = {name: np.ascontiguousarray(value.T) for name, value in (
        ('weak_occupancy_bits', occupancy), ('weak_drivable_bits', drivable),
        ('weak_confidence', cell_confidence), ('weak_drivable_confidence', drive_confidence))}
    if 'ground_labels' in archive:
        ground = archive['ground_labels']
        ground_bits = np.zeros_like(ground)
        ground_bits[ground == 0] = 1
        ground_bits[ground == 1] = 2
        ground_bits[ground == 2] = 4
        result['weak_ground_drivable_bits'] = np.ascontiguousarray(ground_bits.T)
        result['weak_ground_drivable_confidence'] = np.ascontiguousarray(archive['ground_confidence'].T)
    return result


