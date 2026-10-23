"""Check inference imports, optional CUDA execution and official model files."""
import argparse
import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', action='store_true', help='require CUDA, bfloat16 and torchvision CUDA NMS')
    parser.add_argument('--models', action='store_true', help='verify all checkpoint sizes and SHA256')
    args = parser.parse_args(argv)
    checks = {}
    for name in ('numpy', 'scipy', 'cv2', 'lmdb', 'shapely', 'sklearn', 'matplotlib', 'plotly',
                 'PySide6.QtWidgets', 'torch', 'torchvision', 'ultralytics', 'sam3.model_builder',
                 'wilddet3d', 'mdm.model.v2', 'vis4d.op.box.box3d'):
        try:
            importlib.import_module(name)
            checks[name] = 'ok'
        except Exception as exc:
            checks[name] = f'{type(exc).__name__}: {exc}'
    if args.gpu and checks['torch'] == checks['torchvision'] == 'ok':
        try:
            import torch
            from torchvision.ops import nms
            if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
                raise RuntimeError('CUDA with bfloat16 support is required by the current inference recipe')
            value = torch.ones((2, 2), device='cuda', dtype=torch.bfloat16)
            assert (value @ value).float().sum().item() == 8
            nms(torch.tensor([[0., 0., 1., 1.]], device='cuda'), torch.ones(1, device='cuda'), .5)
            torch.cuda.synchronize()
            checks['cuda'] = 'ok'
        except Exception as exc:
            checks['cuda'] = f'{type(exc).__name__}: {exc}'
    if args.models:
        sys.path.insert(0, str(ROOT / 'scripts'))
        from models import main as model_main
        checks['models'] = 'ok' if model_main(['check']) == 0 else 'missing or checksum mismatch'
    print(json.dumps(dict(python=sys.executable, checks=checks), ensure_ascii=False, indent=2))
    return 0 if all(value == 'ok' for value in checks.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
