import numpy as np

CANOPY_PROMPTS = ['overhanging roof', 'metal canopy', 'overhead metal canopy']
PROMPTS = ['roof'] + CANOPY_PROMPTS
THRESHOLDS = {'roof': .4, **dict.fromkeys(CANOPY_PROMPTS, .65)}


def apply_union(old, confidence, evidence):
    votes = sum(evidence[p] > 0 for p in CANOPY_PROMPTS)
    canopy_score = np.maximum.reduce([evidence[p] for p in CANOPY_PROMPTS])
    agreed = votes >= 2
    union_score = np.maximum(evidence['roof'], np.where(agreed, canopy_score, 0))
    selected = (old == 3) & (union_score > 0)
    labels, updated = old.copy(), confidence.copy()
    labels[selected] = 9
    updated[selected] = union_score[selected]
    return labels, updated, selected
