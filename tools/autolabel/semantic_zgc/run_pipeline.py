"""Project-local semantic refinement and projection entry point."""
import json,os,subprocess,sys
from pathlib import Path


def main():
    here=Path(__file__).resolve().parent
    sys.path.insert(0,str(here.parent))
    from runtime_paths import DATA,PROJECT_ROOT
    profile_path=PROJECT_ROOT/'tools/lidar_annotator/configs/zgc.json'
    profile=json.loads(profile_path.read_text())
    for interpreter,name in [(profile['sam_python'],'generate_images.py'),(profile['python'],'project_and_verify.py')]:
        executable=Path(os.environ.get('LIDAR_RUNTIME_PYTHON',interpreter));executable=executable if executable.is_absolute() else profile_path.parent/executable
        subprocess.run([str(executable),str(here/name)],cwd=PROJECT_ROOT,env=os.environ.copy(),check=True)
    print(json.dumps(dict(status='complete',output=str(DATA/'labels'))))


if __name__=='__main__':main()
