"""Run the source deployment, storage and annotation unit checks."""
import argparse,json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
import xml.etree.ElementTree as ET


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--demo',action='store_true');args=parser.parse_args(argv)
    root=Path(__file__).resolve().parents[1];os.chdir(root);sys.path.insert(0,str(root))
    from tools.lidar_annotator.storage import directory
    env=dict(os.environ,QT_QPA_PLATFORM='offscreen',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
    names=['test_cli','test_storage_layout','test_external_storage','test_source_deployment','test_remote_dependencies','test_kitti_input','test_fresh_pipeline','test_public_output','test_incremental_workflow','test_frame_history','test_execution_modes','test_recipe_autosave']
    modules=['tools.lidar_annotator.'+name for name in names]+['test_semantics']
    count=unittest.defaultTestLoader.loadTestsFromNames(modules).countTestCases()
    subprocess.run([sys.executable,'-m','unittest',*modules,'-q'],env=env,check=True)
    tests=['test_person_track_reference.py','test_person_linear_recovery.py','test_person_mask_interior.py','test_fit_vehicle_map.py','test_vehicle_map_dedup.py','test_odom_local_vehicle.py','test_render_combined.py','test_empty_zgc_outputs.py','test_sam3_video_memory.py']
    with tempfile.TemporaryDirectory() as temp:
        report=Path(temp)/'geometry.xml'
        cache=Path(os.environ.get('XDG_CACHE_HOME',Path.home()/'.cache'))/'lidar_annotator/pytest'
        subprocess.run([sys.executable,'-m','pytest',*[str(root/'tools/autolabel/zgc_labeling'/t) for t in tests],'-q','-o','cache_dir='+str(cache),'--junitxml='+str(report)],env=env,check=True)
        geometry=sum(int(node.attrib.get('tests',0)) for node in ET.parse(report).getroot().iter('testsuite'))
    if args.demo:subprocess.run([sys.executable,str(root/'lidar_annotator.py'),'demo'],env=env,check=True)
    result=dict(status='passed',unit_tests=count,geometry_tests=geometry,model_demo=args.demo,live_ui_checked=False)
    destination=directory('verification');destination.mkdir(parents=True,exist_ok=True)
    (destination/'self_test.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
