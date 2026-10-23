"""Source release and download contracts, without real model downloads."""
import hashlib
import io
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from scripts import export_source, models
from tools.autolabel.resource_paths import dependency_root
from .test_remote_dependencies import launcher_fixture

ROOT = Path(__file__).resolve().parents[2]


class ModelDeploymentTests(unittest.TestCase):
    def record(self, data):
        return dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())

    def test_download_is_verified_and_valid_target_reused(self):
        source = dict(url='https://example.invalid/weights', homepage='https://example.invalid')
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            with patch('urllib.request.urlopen', return_value=io.BytesIO(b'official')) as fetch:
                models.download('model.pt', source, self.record(b'official'), cache)
                models.download('model.pt', source, self.record(b'official'), cache)
            fetch.assert_called_once()
            self.assertEqual((cache / 'model.pt').read_bytes(), b'official')

    def test_wrong_download_preserves_existing_target(self):
        source = dict(url='https://example.invalid/weights', homepage='https://example.invalid')
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory); target = cache / 'model.pt'; target.write_bytes(b'previous')
            with patch('urllib.request.urlopen', return_value=io.BytesIO(b'incorrect')):
                with self.assertRaisesRegex(ValueError, 'SHA256'):
                    models.download('model.pt', source, self.record(b'official'), cache)
            self.assertEqual(target.read_bytes(), b'previous')
            self.assertFalse((cache / 'model.pt.downloading').exists())

    def test_same_size_corruption_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pt'; path.write_bytes(b'corrupt!')
            self.assertFalse(models.verified(path, self.record(b'official')))

    def test_huggingface_download_uses_pinned_revision(self):
        source = dict(repo_id='official/model', filename='weights.pt', revision='fixed-revision',
                      homepage='https://example.invalid')
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory); upstream = cache / 'hf-cache'; upstream.write_bytes(b'official')
            with patch.dict(sys.modules, huggingface_hub=type('Hub', (), {
                'hf_hub_download': staticmethod(lambda **kw: self.assert_hf(kw, upstream))})):
                models.download('model.pt', source, self.record(b'official'), cache)
            self.assertTrue(models.verified(cache / 'model.pt', self.record(b'official')))

    def assert_hf(self, arguments, upstream):
        self.assertEqual(arguments, dict(repo_id='official/model', filename='weights.pt', revision='fixed-revision'))
        return upstream


class SourceReleaseTests(unittest.TestCase):
    def test_samples_are_a_separate_archive_without_labels_or_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);samples=root/'samples'
            for name in ('examples/kitti_minimal_two_frames/image_2/000000.png',
                         'testdata/lmdb_sample/index.lmdb/data.mdb',
                         'examples/kitti_minimal_two_frames/labels/000000.label',
                         'examples/kitti_minimal_two_frames/model.pt'):
                path=samples/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'sample')
            output=root/'samples.tar.gz';export_source.build_samples(samples,output)
            with tarfile.open(output) as archive:
                names={m.name for m in archive.getmembers()}
            self.assertIn('lidar-annotator-samples/examples/kitti_minimal_two_frames/image_2/000000.png',names)
            self.assertFalse(any('/labels/' in name or name.endswith('.pt') for name in names))

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'requires the application environment')
    def test_sam_image_and_frame_inference_do_not_require_the_training_video_decoder(self):
        code = '''
import builtins
original = builtins.__import__
def without_decord(name, *args, **kwargs):
    if name == 'decord' or name.startswith('decord.'):
        raise ModuleNotFoundError('training-only decord is unavailable')
    return original(name, *args, **kwargs)
builtins.__import__ = without_decord
import sam3.model_builder
'''
        env = {**os.environ, 'PYTHONPATH': str(dependency_root() / 'sam3')}
        result = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless(importlib.util.find_spec('PySide6'), 'requires the application environment')
    def test_source_opencv_qt5_paths_are_removed_before_qt6_starts(self):
        from tools.lidar_annotator import qt_app
        class ReachedQt(Exception):
            pass
        def inspect_environment():
            self.assertFalse('QT_QPA_PLATFORM_PLUGIN_PATH' in os.environ)
            self.assertFalse('QT_QPA_FONTDIR' in os.environ)
            raise ReachedQt
        with patch.dict(os.environ, {'QT_QPA_PLATFORM_PLUGIN_PATH': '/env/site-packages/cv2/qt/plugins',
                                    'QT_QPA_FONTDIR': '/env/site-packages/cv2/qt/fonts'}):
            with patch.object(qt_app, 'QApplication') as app:
                app.instance.side_effect = inspect_environment
                with self.assertRaises(ReachedQt):
                    qt_app.main([])
        def inspect_custom_paths():
            self.assertEqual(os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'], '/custom/qt/plugins')
            self.assertEqual(os.environ['QT_QPA_FONTDIR'], '/custom/fonts')
            raise ReachedQt
        with patch.dict(os.environ, {'QT_QPA_PLATFORM_PLUGIN_PATH': '/custom/qt/plugins',
                                    'QT_QPA_FONTDIR': '/custom/fonts'}):
            with patch.object(qt_app, 'QApplication') as app:
                app.instance.side_effect = inspect_custom_paths
                with self.assertRaises(ReachedQt):
                    qt_app.main([])

    @unittest.skipUnless(shutil.which('git'), 'git is needed to check publication ignore rules')
    def test_gitignore_keeps_application_and_vendor_data_modules_publishable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); shutil.copyfile(ROOT / '.gitignore', root / '.gitignore')
            subprocess.run(['git', 'init', '-q', str(root)], check=True)
            names = ['tools/lidar_annotator/fresh_pipeline.py',
                     'third_party/wilddet3d/wilddet3d/data/transforms/resize.py',
                     'data/private.bin', 'runtime/python.tar', 'lidar_annotator/private.py', 'models/sam3.pt']
            result = subprocess.run(['git', '-C', str(root), 'check-ignore', '--no-index', *names],
                                    capture_output=True, text=True)
            self.assertEqual(set(result.stdout.splitlines()), set(names[2:]))

    def test_current_source_selection_covers_runtime_modules_and_excludes_local_artifacts(self):
        names = {p.relative_to(ROOT).as_posix() for p in export_source.selected_files(ROOT)}
        for name in ('tools/lidar_annotator/fresh_pipeline.py', 'tools/lidar_annotator/kitti_input.py',
                     'tools/lidar_annotator/dataset_labels.py', 'tools/autolabel/zgc_labeling/finalize_person_positions.py',
                     'third_party/patches/sam3.patch', 'third_party/patches/wilddet3d.patch', 'scripts/setup_env.sh'):
            self.assertIn(name, names)
        self.assertFalse(any(n.startswith(('runtime/', 'outputs/', 'lidar_annotator/','testdata/','examples/')) for n in names))
        self.assertFalse(any(n.endswith(('.pt', '.pyc', '.so')) for n in names))
        self.assertNotIn('configs/paths.json', names)
        self.assertNotIn('configs/user_recipe.json', names)

    def test_archive_manifest_matches_bytes_and_configs_are_portable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'source'; root.mkdir()
            # Only source files and portable configuration are published.
            for name in export_source.ROOT_FILES:
                path = root / name; path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / name, path)
            for name in ('tools/new_module.py', 'runtime/huge.tar', 'models/sam3.pt',
                         'lidar_annotator/private.py', 'tools/__pycache__/x.pyc','testdata/private.bin','examples/private.bin'):
                path = root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b'test')
            output = Path(directory) / 'release.tar.gz'; export_source.build(root, output)
            with tarfile.open(output) as archive:
                prefix = 'lidar-annotator-source/'
                data = {m.name[len(prefix):]: archive.extractfile(m).read() for m in archive.getmembers()}
            manifest = json.loads(data.pop('source_manifest.json'))['files']
            self.assertEqual(set(manifest), set(data))
            for name, content in data.items():
                self.assertEqual(manifest[name], dict(bytes=len(content), sha256=hashlib.sha256(content).hexdigest()))
            self.assertIn('tools/new_module.py', data)
            self.assertNotIn('models/sam3.pt', data)
            self.assertNotIn('runtime/huge.tar', data)
            self.assertNotIn('testdata/private.bin',data)
            self.assertNotIn('examples/private.bin',data)
            self.assertNotIn('configs/paths.json',data)
            self.assertNotIn(b'/mnt/',data['configs/paths.source.json'])

    def test_source_launcher_passes_custom_python_and_portable_paths_without_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); shutil.copyfile(ROOT / 'run_source.sh', root / 'run_source.sh')
            launcher_fixture(root,root/'cache')
            (root / 'lidar_annotator.py').write_text(
                'import json,os,sys; print(json.dumps([sys.argv[1:],os.environ["LIDAR_ANNOTATOR_PATHS"]]))')
            env = {**os.environ, 'LIDAR_PYTHON': sys.executable,'XDG_CACHE_HOME':str(root/'cache')}
            env.pop('LIDAR_ANNOTATOR_PATHS', None)
            result = subprocess.run(['bash', str(root / 'run_source.sh'), 'annotate', '--help'],
                                    env=env, check=True, capture_output=True, text=True)
            arguments, paths = json.loads(result.stdout)
            self.assertEqual(arguments, ['annotate', '--help'])
            self.assertEqual(paths, str(root / 'configs/paths.source.json'))


if __name__ == '__main__':
    unittest.main()
