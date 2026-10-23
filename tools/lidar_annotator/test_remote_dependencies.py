"""Remote resources are pinned, verified, safely unpacked and locally patched."""
import difflib
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from scripts import dependencies

ROOT=Path(__file__).resolve().parents[2]


def sha(data):return hashlib.sha256(data).hexdigest()


def tar_bytes(files):
    stream=io.BytesIO()
    with tarfile.open(fileobj=stream,mode='w:gz') as archive:
        for name,content in files.items():
            info=tarfile.TarInfo(name);info.size=len(content);archive.addfile(info,io.BytesIO(content))
    return stream.getvalue()


def launcher_fixture(root,cache_home):
    """Small valid resource cache for launcher tests, with no network or models."""
    for name in ('scripts/dependencies.py','tools/autolabel/resource_paths.py'):
        target=root/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    for name in ('tools/__init__.py','tools/autolabel/__init__.py'):(root/name).write_text('')
    lock=root/'configs/dependencies.lock.json';lock.parent.mkdir(exist_ok=True)
    lock.write_text(json.dumps(dict(schema='lidar_remote_dependencies_v1',sources=[],assets=[])))
    destination=dependencies.dependency_root(lock,cache_home/'lidar_annotator/dependencies');destination.mkdir(parents=True)
    (destination/'payload.txt').write_bytes(b'fixture')
    (destination/'ready.json').write_text(json.dumps(dict(lock_sha256=dependencies.digest(lock),files={'payload.txt':sha(b'fixture')})))


class RemoteDependencyTests(unittest.TestCase):
    def fixture(self,root,archive):
        project=root/'project';project.mkdir();lock=project/'lock.json'
        source=dict(name='source',repository='https://github.com/example/source',commit='1'*40,
                    url='https://example.invalid/source',sha256=sha(archive),target='source',include=['package','LICENSE'])
        font=dict(name='font',repository='https://github.com/example/font',commit='2'*40,
                  url='https://example.invalid/font',sha256=sha(b'font'),target='fonts/font.ttf')
        lock.write_text(json.dumps(dict(schema='lidar_remote_dependencies_v1',sources=[source],assets=[font])))
        return project,lock,root/'cache'

    def test_fetch_verifies_and_reuses_exact_cache_and_detects_changes(self):
        archive=tar_bytes({'repo/package/value.py':b'value=1\n','repo/LICENSE':b'license'})
        with tempfile.TemporaryDirectory() as temp:
            project,lock,cache=self.fixture(Path(temp),archive)
            with patch('urllib.request.urlopen',side_effect=[io.BytesIO(archive),io.BytesIO(b'font')]) as fetch:
                result=dependencies.fetch(lock,cache,project)
                self.assertTrue(dependencies.check(lock,cache,project))
                self.assertEqual(dependencies.fetch(lock,cache,project),result)
                self.assertEqual(fetch.call_count,2)
            (result/'source/package/value.py').write_text('changed')
            self.assertFalse(dependencies.check(lock,cache,project))
            with self.assertRaisesRegex(ValueError,'modified'):dependencies.fetch(lock,cache,project)
            self.assertEqual((result/'source/package/value.py').read_text(),'changed')
            (result/'source/package/value.py').write_text('value=1\n')
            (result/'source/package/untracked.py').write_text('extra')
            self.assertFalse(dependencies.check(lock,cache,project))

    def test_bad_download_never_creates_ready_cache(self):
        archive=tar_bytes({'repo/package/value.py':b'value=1\n','repo/LICENSE':b'license'})
        with tempfile.TemporaryDirectory() as temp:
            project,lock,cache=self.fixture(Path(temp),archive)
            with patch('urllib.request.urlopen',return_value=io.BytesIO(b'corrupt')):
                with self.assertRaisesRegex(ValueError,'SHA256'):dependencies.fetch(lock,cache,project)
            self.assertFalse(dependencies.dependency_root(lock,cache).exists())
            self.assertFalse(list(cache.rglob('*.downloading')))

    def test_archive_cannot_escape_destination(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);archive=root/'bad.tar.gz';archive.write_bytes(tar_bytes({'repo/../escape.py':b'bad'}))
            with self.assertRaisesRegex(ValueError,'Unsafe'):dependencies.extract_sources(archive,root/'target',['package'])
            self.assertFalse((root/'escape.py').exists())

    def test_archive_symlinks_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path=root/'bad.tar.gz'
            with tarfile.open(path,'w:gz') as archive:
                info=tarfile.TarInfo('repo/package/link');info.type=tarfile.SYMTYPE;info.linkname='/tmp/outside';archive.addfile(info)
            with self.assertRaisesRegex(ValueError,'Unsupported'):dependencies.extract_sources(path,root/'target',['package'])

    def test_patches_require_exact_upstream_and_patched_bytes(self):
        before=b'value=1\n';after=b'value=2\n'
        archive=tar_bytes({'repo/package/value.py':before,'repo/LICENSE':b'license'})
        with tempfile.TemporaryDirectory() as temp:
            project,lock,cache=self.fixture(Path(temp),archive)
            text=''.join(difflib.unified_diff(before.decode().splitlines(keepends=True),after.decode().splitlines(keepends=True),fromfile='a/package/value.py',tofile='b/package/value.py'))
            (project/'change.patch').write_text(text);data=json.loads(lock.read_text())
            data['sources'][0]['patches']=[dict(file='change.patch',sha256=sha(text.encode()),files=[dict(file='package/value.py',upstream_sha256=sha(before),patched_sha256=sha(after))])]
            lock.write_text(json.dumps(data))
            with patch('urllib.request.urlopen',side_effect=[io.BytesIO(archive),io.BytesIO(b'font')]):
                result=dependencies.fetch(lock,cache,project)
            self.assertEqual((result/'source/package/value.py').read_bytes(),after)
            (project/'change.patch').write_text('modified')
            self.assertFalse(dependencies.check(lock,cache,project))

    def test_project_lock_has_full_immutable_github_commits(self):
        lock=dependencies.read_lock(ROOT/'configs/dependencies.lock.json')
        self.assertEqual(len(lock['sources']),7)
        for source in lock['sources']:
            self.assertIn(source['commit'],source['url'])
            self.assertEqual(len(source['commit']),40)
            for item in source.get('patches',[]):self.assertEqual(dependencies.digest(ROOT/item['file']),item['sha256'])


if __name__=='__main__':unittest.main()
