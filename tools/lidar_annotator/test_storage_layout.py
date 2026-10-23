"""Cached resources must work without model binaries or a runtime in the source tree."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts import bootstrap_models
from .test_remote_dependencies import launcher_fixture

ROOT = Path(__file__).resolve().parents[2]


class StorageLayoutTests(unittest.TestCase):
    def test_direct_import_uses_portable_path_config(self):
        import sys
        environment=dict(os.environ);environment.pop('LIDAR_ANNOTATOR_PATHS',None)
        result=subprocess.run([sys.executable,'-c','from tools.autolabel.runtime_paths import CONFIG; print(CONFIG.name)'],cwd=ROOT,env=environment,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(result.stdout.strip(),'paths.source.json')

    def test_source_environment_is_loaded_from_user_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);root=base/'source';root.mkdir()
            shutil.copyfile(ROOT/'run_source.sh',root/'run_source.sh')
            (root/'lidar_annotator.py').write_text('import sys; print(" ".join(sys.argv[1:]))')
            python=base/'cache/lidar_annotator/venvs/default/bin/python';python.parent.mkdir(parents=True)
            launcher_fixture(root,base/'cache')
            import sys
            python.symlink_to(sys.executable)
            environment={**os.environ,'XDG_CACHE_HOME':str(base/'cache')};environment.pop('LIDAR_PYTHON',None)
            result=subprocess.run(['bash',str(root/'run_source.sh'),'annotate','--help'],env=environment,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(result.stdout.strip(),'annotate --help')
            self.assertFalse((root/'.venv').exists())

    def test_cached_runtime_launches_without_a_project_runtime_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);root=base/'source';(root/'configs').mkdir(parents=True)
            shutil.copyfile(ROOT/'run.sh',root/'run.sh');(root/'configs/runtime-id').write_text('test-runtime\n')
            (root/'run_source.sh').write_text('#!/bin/bash\nexec "$LIDAR_PYTHON" "'+str(root/'lidar_annotator.py')+'" "$@"\n')
            runtime=base/'cache/runtime/test-runtime';(runtime/'bin').mkdir(parents=True)
            (runtime/'.install-prefix').write_text(str(runtime))
            python=runtime/'bin/python'
            python.write_text('#!/bin/bash\nif [[ "$1" == */bootstrap_models.py ]]; then exit 0; fi\nprintf "%s\\n" "$@"\n')
            python.chmod(0o755)
            result=subprocess.run(['bash',str(root/'run.sh'),'--help'],capture_output=True,text=True,
                                  env={**os.environ,'LIDAR_RUNTIME_CACHE':str(base/'cache/runtime'),
                                       'LIDAR_OFFLINE_BUNDLE':str(base/'cache/bundle'),'XDG_CACHE_HOME':str(base/'cache'),'LIDAR_PYTHON':''})
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(result.stdout.splitlines(),[str(root/'lidar_annotator.py'),'--help'])
            self.assertFalse((root/'runtime').exists())

    def test_metadata_only_model_directory_reuses_existing_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'source';source=root/'models';source.mkdir(parents=True)
            cache=Path(directory)/'cache';cache.mkdir();(cache/'sam3.pt').write_bytes(b'model')
            (source/'manifest.json').write_text(json.dumps({'sam3.pt':dict(bytes=5,sha256=hashlib.sha256(b'model').hexdigest())}))
            (source/'download_manifest.json').write_text('{}')
            with patch.object(bootstrap_models,'__file__',str(root/'scripts/bootstrap_models.py')),patch.dict(os.environ,{'LIDAR_MODEL_CACHE':str(cache)}):
                bootstrap_models.main()
            self.assertEqual((cache/'sam3.pt').read_bytes(),b'model')
            self.assertTrue((cache/'manifest.json').is_file())

    def test_missing_cached_model_explains_how_to_download_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'source';source=root/'models';source.mkdir(parents=True)
            (source/'manifest.json').write_text(json.dumps({'sam3.pt':dict(bytes=5,sha256='missing')}))
            with patch.object(bootstrap_models,'__file__',str(root/'scripts/bootstrap_models.py')),patch.dict(os.environ,{'LIDAR_MODEL_CACHE':str(Path(directory)/'cache')}):
                with self.assertRaisesRegex(FileNotFoundError,'models download'):
                    bootstrap_models.main()


if __name__=='__main__':
    unittest.main()
