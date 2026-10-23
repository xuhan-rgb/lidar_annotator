import numpy as np


def training_image_labels(raw_labels):
    assert set(np.unique(raw_labels)) <= set(range(10)) | {255}
    labels=raw_labels.copy()
    labels[np.isin(labels,(7,8))]=6
    labels[labels==9]=4
    return labels
