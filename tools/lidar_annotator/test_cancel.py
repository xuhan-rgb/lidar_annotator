"""Verify cancellation terminates an executing backend, rather than orphaning it."""
import os,signal,subprocess,sys,tempfile,time,unittest
from pathlib import Path
from .cli import REPO


class CancellationTests(unittest.TestCase):
    def test_stop_is_quiet_at_cli_and_worker_boundaries(self):
        for worker in (False,True):
            with self.subTest(worker=worker),tempfile.TemporaryDirectory() as tmp:
                ready=Path(tmp)/'ready';script=Path(tmp)/'entry.py'
                command='annotate-stage' if worker else 'annotate'
                arguments=['--job',tmp,'--stage','semantic'] if worker else ['--source',tmp,'--output',tmp]
                code=(f"import sys,runpy,time;from pathlib import Path;sys.path.insert(0,{str(REPO)!r});"
                      "from tools.lidar_annotator import fresh_pipeline as pipeline\n"
                      f"def wait(*args,**kwargs):\n Path({str(ready)!r}).touch()\n time.sleep(60)\n"
                      f"pipeline.{'stage' if worker else 'run'}=wait\n"
                      f"sys.argv={[str(REPO/'lidar_annotator.py'),command,*arguments]!r}\n"
                      f"runpy.run_path({str(REPO/'lidar_annotator.py')!r},run_name='__main__')\n")
                script.write_text(code);proc=subprocess.Popen([sys.executable,str(script)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
                try:
                    deadline=time.monotonic()+10
                    while not ready.exists() and proc.poll() is None and time.monotonic()<deadline:time.sleep(.01)
                    self.assertTrue(ready.exists());proc.terminate();out,err=proc.communicate(timeout=10)
                    self.assertEqual(proc.returncode,130);self.assertNotIn('Traceback',out+err);self.assertNotIn('KeyboardInterrupt',out+err)
                    if not worker:self.assertIn('已停止',out+err)
                finally:
                    if proc.poll() is None:proc.kill();proc.communicate()

    def test_genuine_error_still_reports_failure(self):
        code=(f"import sys,runpy;sys.path.insert(0,{str(REPO)!r});"
              "from tools.lidar_annotator import fresh_pipeline as pipeline;"
              "pipeline.stage=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('genuine inference failure'));"
              f"sys.argv={[str(REPO/'lidar_annotator.py'),'annotate-stage','--job','/tmp/unused','--stage','semantic']!r};"
              f"runpy.run_path({str(REPO/'lidar_annotator.py')!r},run_name='__main__')")
        result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
        self.assertEqual(result.returncode,1);self.assertIn('genuine inference failure',result.stderr);self.assertIn('Traceback',result.stderr)

    def test_terminate_reaps_backend_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            pidfile=Path(tmp)/'child.pid';script=Path(tmp)/'worker.py'
            child=f"import os,time;from pathlib import Path;Path({str(pidfile)!r}).write_text(str(os.getpid()));time.sleep(60)"
            script.write_text(f"import sys;sys.path.insert(0,{str(REPO)!r});from tools.lidar_annotator.cli import execute;execute([sys.executable,'-c',{child!r}],check=True)")
            proc=subprocess.Popen([sys.executable,str(script)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            try:
                deadline=time.monotonic()+10
                while not pidfile.exists() and time.monotonic()<deadline:time.sleep(.01)
                self.assertTrue(pidfile.exists());child_pid=int(pidfile.read_text());proc.send_signal(signal.SIGTERM);proc.wait(timeout=10)
                with self.assertRaises(ProcessLookupError):os.kill(child_pid,0)
            finally:
                if proc.poll() is None:proc.kill();proc.wait()


if __name__=='__main__':unittest.main()
