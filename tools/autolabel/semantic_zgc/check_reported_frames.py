import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

folder = Path(sys.argv[1])
from common import SOURCE
base = SOURCE/'zgc'
report = {'scope': 'eight selected regression frames; not ground-truth accuracy measurement'}
for frame in ('000000', '000012', '000017', '000021', '000221', '000298', '000411', '000440'):
    old = np.array(Image.open(base / 'image_masks' / (frame + '.png')))
    new = np.array(Image.open(folder / 'image_masks' / (frame + '.png')))
    np.testing.assert_array_equal(new[old != 3], old[old != 3])
    report[frame] = {'roof_pixels': int((new == 9).sum()), 'added_pixels': int(((old == 3) & (new == 9)).sum()),
                     'nonbuilding_labels_preserved': True}
assert report['000012']['added_pixels'] > 100000, report
assert report['000000']['added_pixels'] > 100000, report
print(json.dumps(report, indent=2))
