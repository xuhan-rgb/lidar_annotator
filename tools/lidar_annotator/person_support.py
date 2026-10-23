"""Conservative selection of semantic person points for an instance."""

import numpy as np


def select_person_support(xyz, support_ids, labels, center, saved_ids=None, radius=0.5):
    """Return person support indices to keep or reject, without changing inputs."""
    xyz = np.asarray(xyz)
    support = np.asarray(support_ids, dtype=np.int64)
    person = support[np.asarray(labels)[support] == 1] if labels is not None else support.copy()
    person = np.asarray(person, dtype=np.int64)

    def result(kept, rejected, status, review, center_xy=None, saved=False):
        return dict(kept_indices=np.asarray(kept, dtype=np.int64),
                    rejected_indices=np.asarray(rejected, dtype=np.int64),
                    status=status, review_required=bool(review), center_xy=center_xy,
                    saved_support_used=bool(saved))

    if labels is None or center is None:
        return result(person, [], 'missing_labels_or_center', True)
    try:
        center_arr = np.asarray(center, dtype=float).reshape(-1)
        if center_arr.size < 2 or not np.isfinite(center_arr[:2]).all():
            return result(person, [], 'invalid_center', True)
    except (TypeError, ValueError):
        return result(person, [], 'invalid_center', True)
    center_xy = center_arr[:2]

    saved = np.empty(0, dtype=np.int64)
    if saved_ids is not None:
        raw = np.asarray(saved_ids).reshape(-1)
        try:
            numeric = raw.astype(float)
            if (not np.isfinite(numeric).all() or np.any(numeric < 0) or
                    np.any(numeric != np.floor(numeric)) or np.any(numeric >= len(xyz))):
                return result(person, [], 'invalid_saved_support', True, center_xy.tolist())
            saved = numeric.astype(np.int64)
        except (TypeError, ValueError, OverflowError):
            return result(person, [], 'invalid_saved_support', True, center_xy.tolist())
        saved = np.intersect1d(saved, person, assume_unique=False)

    if len(person) < 5:
        return result(person, [], 'fewer_than_5_person_points', True, center_xy.tolist())
    xy = np.asarray(xyz[person, :2], dtype=float)
    if not np.isfinite(xy).all():
        return result(person, [], 'invalid_person_xy', True, center_xy.tolist())
    initial = np.median(xy, axis=0)
    core_mask = np.linalg.norm(xy - initial, axis=1) <= radius
    core = person[core_mask]
    if len(core) < 5 or len(core) < 0.5 * len(person):
        # Uncertain local structure must not preserve arbitrarily distant returns.
        # This bound covers the existing core radius plus its allowed centre shift.
        maximum_distance = radius + 0.75
        nearby = person[np.linalg.norm(xy - center_xy, axis=1) <= maximum_distance]
        kept = np.union1d(nearby, saved)
        rejected = np.setdiff1d(person, kept, assume_unique=False)
        decision = result(kept, rejected, 'insufficient_person_core_far_points_ignored' if len(rejected) else 'insufficient_person_core',
                          True, center_xy.tolist(), bool(len(saved)))
        decision['fallback_max_distance_m'] = maximum_distance
        return decision
    core_center = np.median(np.asarray(xyz[core, :2], dtype=float), axis=0)
    if np.linalg.norm(core_center - center_xy) > 0.75:
        return result(person, [], 'person_core_center_mismatch', True, center_xy.tolist())
    keep_mask = np.linalg.norm(xy - core_center, axis=1) <= radius
    kept = np.union1d(person[keep_mask], saved)
    rejected = np.setdiff1d(person, kept, assume_unique=False)
    return result(kept, rejected, 'person_support_filtered', False,
                  core_center.tolist(), bool(len(saved)))
