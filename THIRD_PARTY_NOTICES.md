# Third-party materials

Remote source revisions, archive SHA256 values and project patch hashes are pinned
in `configs/dependencies.lock.json`. The source trees and fonts are downloaded to
`~/.cache/lidar_annotator/dependencies/<lock-id>/`; they are not vendored in this repository.
The cache includes upstream license files. Copies of the relevant notices remain
under `third_party/licenses/`, particularly for code represented in local patches.

- Meta SAM3 and the WildDet3D SAM3 fork: SAM License; `third_party/licenses/sam3.txt` and `sam3-for-wilddet3d.txt`.
- Ai2 WildDet3D: SAM License; `third_party/licenses/wilddet3d.txt`.
- Robbyant LingBot-Depth: `third_party/licenses/lingbot-depth.txt`.
- EasternJournalist utils3d: MIT; `third_party/licenses/utils3d.txt`.
- SysCV Vis4D 1.0.0: Apache-2.0; `third_party/licenses/vis4d.txt`.
- msgpack-python 1.0.3: `third_party/licenses/msgpack.txt`.
- Noto Sans CJK SC: `third_party/licenses/Noto.txt`.
- DejaVu Sans: `third_party/licenses/DejaVu.txt`.
- termcolor, ml-collections and absl-py are installed from pinned PyPI versions;
  their distribution metadata and licenses are retained in the installed environment.
- PySide6, Python, PyTorch, CUDA user libraries and other installed dependencies
  retain their own distribution licenses. The system NVIDIA driver is not distributed.

The local SAM3 and WildDet3D changes are maintained in `third_party/patches/`.
Their exact input and output hashes are part of the lock file. See
`docs/remote_dependencies.md` for the source mapping and update procedure.
Original annotation-module hashes are recorded in `docs/provenance.json`.
The application's own SVG icon remains under `assets/icons/`.
