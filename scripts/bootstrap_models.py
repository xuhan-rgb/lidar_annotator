"""Seed the per-user model cache from the offline files shipped with this project."""
import hashlib,json,os,shutil
from pathlib import Path


def main():
    root=Path(__file__).resolve().parents[1];source=root/'models'
    cache=Path(os.environ.get('LIDAR_MODEL_CACHE',Path.home()/'.cache/lidar_annotator/models')).expanduser()
    manifest=json.loads((source/'manifest.json').read_text());cache.mkdir(parents=True,exist_ok=True)
    for name,record in manifest.items():
        dest=cache/name
        if dest.is_file() and dest.stat().st_size==record['bytes']:continue
        if not (source/name).is_file():
            raise FileNotFoundError(f'Model missing from cache: {dest}. Run bash run_source.sh models download, or run scripts/models.py download with the runtime Python.')
        print(f'Preparing offline model cache: {name}',flush=True);temp=dest.with_suffix('.copying')
        shutil.copy2(source/name,temp)
        with temp.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        if digest!=record['sha256']:temp.unlink();raise ValueError(f'Model checksum failed: {name}')
        temp.replace(dest)
    for name in ['manifest.json','download_manifest.json']:
        if cache.resolve()!=source.resolve():shutil.copy2(source/name,cache/name)


if __name__=='__main__':main()
