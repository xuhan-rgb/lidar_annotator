"""Compact, lossless storage for per-class semantic planes."""
import hashlib
import json
import os
from pathlib import Path
from zipfile import BadZipFile

import numpy as np


def _receipt_path(path):
    return Path(path).with_suffix(Path(path).suffix + '.json')


def _receipt_digest(receipt):
    unsigned = {key: value for key, value in receipt.items() if key != 'receipt_sha256'}
    canonical = json.dumps(unsigned, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(canonical).hexdigest()


def _bbox(planes, shape):
    occupied = np.zeros(shape, dtype=bool)
    for plane in planes.values():
        if plane.dtype == bool:
            occupied |= plane
        else:
            occupied |= np.any(plane.view(np.uint8).reshape(*shape, 4) != 0, axis=2)
    ys, xs = np.nonzero(occupied)
    if not len(xs):
        return None
    return [int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1]


def save_class_cache(path, planes, metadata):
    path = Path(path)
    planes = dict(planes)
    shape = None
    for name, plane in planes.items():
        if not isinstance(name, str):
            raise TypeError('class names must be strings')
        plane = np.asarray(plane)
        if plane.dtype not in (np.dtype(np.float32), np.dtype(bool)) or plane.ndim != 2:
            raise TypeError('planes must be 2D float32 or bool arrays')
        if shape is None:
            shape = plane.shape
        elif plane.shape != shape:
            raise ValueError('all planes must have the same shape')
        planes[name] = plane
    if shape is None:
        raise ValueError('at least one plane is required')

    bbox = _bbox(planes, shape)
    if bbox is None:
        crop_shape = (0, 0)
        y0 = x0 = 0
    else:
        y0, x0, y1, x1 = bbox
        crop_shape = (y1 - y0, x1 - x0)
    arrays = {}
    descriptors = {}
    for index, (name, plane) in enumerate(planes.items()):
        key = 'p%d' % index
        descriptors[name] = {'key': key, 'dtype': 'bool' if plane.dtype == bool else 'float32'}
        crop = plane[y0:y0 + crop_shape[0], x0:x0 + crop_shape[1]]
        if plane.dtype == bool:
            arrays[key] = np.packbits(crop.reshape(-1), bitorder='little')
        else:
            raw = np.ascontiguousarray(crop, dtype=np.float32).view(np.uint8).reshape(-1, 4)
            arrays[key] = raw.T.copy()

    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    digest = hashlib.sha256(temp.read_bytes()).hexdigest()
    os.replace(temp, path)
    receipt = {'sha256': digest, 'shape': list(shape), 'bbox': bbox,
               'planes': descriptors, 'metadata': metadata}
    receipt['receipt_sha256'] = _receipt_digest(receipt)
    receipt_path = _receipt_path(path)
    receipt_temp = receipt_path.with_name(receipt_path.name + '.tmp')
    receipt_temp.write_text(json.dumps(receipt, separators=(',', ':')))
    os.replace(receipt_temp, receipt_path)


def load_class_cache(path, shape):
    path = Path(path)
    try:
        receipt = json.loads(_receipt_path(path).read_text())
        if _receipt_digest(receipt) != receipt['receipt_sha256']:
            return None
        if tuple(receipt['shape']) != tuple(shape):
            return None
        if hashlib.sha256(path.read_bytes()).hexdigest() != receipt['sha256']:
            return None
        full_shape = tuple(receipt['shape'])
        bbox = receipt['bbox']
        if bbox is None:
            y0 = x0 = 0
            crop_shape = (0, 0)
        else:
            y0, x0, y1, x1 = bbox
            crop_shape = (y1 - y0, x1 - x0)
        planes = {}
        with np.load(path, allow_pickle=False) as archive:
            for name, descriptor in receipt['planes'].items():
                crop_size = crop_shape[0] * crop_shape[1]
                if descriptor['dtype'] == 'bool':
                    packed = archive[descriptor['key']]
                    crop = np.unpackbits(packed, bitorder='little', count=crop_size).astype(bool)
                elif descriptor['dtype'] == 'float32':
                    shuffled = archive[descriptor['key']]
                    crop = shuffled.T.copy().reshape(-1).view(np.float32)
                else:
                    return None
                if crop.size != crop_size:
                    return None
                plane = np.zeros(full_shape, dtype=bool if descriptor['dtype'] == 'bool' else np.float32)
                if bbox is not None:
                    plane[y0:y0 + crop_shape[0], x0:x0 + crop_shape[1]] = crop.reshape(crop_shape)
                planes[name] = plane
        return planes, receipt['metadata']
    except (OSError, ValueError, TypeError, KeyError, BadZipFile, json.JSONDecodeError):
        return None
