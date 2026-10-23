"""Inspect Python, GPU availability and commit-locked cached model source."""
import json,platform,sys
from pathlib import Path


def main():
    import numpy,scipy,cv2,lmdb,shapely,sklearn,torch
    import PySide6
    import sam3.model_builder
    root=Path(__file__).resolve().parents[2]
    result=dict(platform=platform.platform(),architecture=platform.machine(),glibc=platform.libc_ver(),python=sys.executable,torch=torch.__version__,cuda_runtime=torch.version.cuda,cuda_available=torch.cuda.is_available(),qt=PySide6.__version__,sam_source=sam3.model_builder.__file__,project=str(root),nvidia_driver_required=True)
    lock=json.loads((root/'configs/dependencies.lock.json').read_text())
    result['dependency_commits']={item['name']:item['commit'] for item in lock['sources']}
    if torch.cuda.is_available():
        tensor=torch.ones(4,device='cuda')*2;torch.cuda.synchronize();result.update(gpu=torch.cuda.get_device_name(0),cuda_tensor_check=bool(tensor.sum().item()==8))
    print(json.dumps(result,ensure_ascii=False,indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
