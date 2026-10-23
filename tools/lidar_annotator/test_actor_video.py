import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import actor_video as av


class ActorVideoTests(unittest.TestCase):
    def test_vendor_propagation_does_not_print_progress(self):
        import ast,contextlib,io
        from types import SimpleNamespace
        from tqdm import tqdm
        source=Path(__file__).resolve().parents[2]/'third_party/sam3/sam3/model/sam3_video_inference.py'
        tree=ast.parse(source.read_text())
        calls=[node for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='tqdm' and any(k.arg=='desc' and isinstance(k.value,ast.Constant) and k.value.value=='propagate_in_video' for k in node.keywords)]
        self.assertEqual(len(calls),1)
        captured=io.StringIO()
        with contextlib.redirect_stderr(captured):
            result=list(eval(compile(ast.Expression(calls[0]),str(source),'eval'),{'tqdm':tqdm,'processing_order':range(10),'self':SimpleNamespace(rank=0)}))
        self.assertEqual(result,list(range(10)))
        self.assertEqual(captured.getvalue(),'')

    def test_image_folder_loading_does_not_print_progress(self):
        import ast,contextlib,io
        from types import SimpleNamespace
        from tqdm import tqdm
        source=Path(__file__).resolve().parents[2]/'third_party/sam3/sam3/model/io_utils.py'
        tree=ast.parse(source.read_text())
        calls=[node for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='tqdm' and any(k.arg=='desc' and 'frame loading (image folder)' in ast.unparse(k.value) for k in node.keywords)]
        self.assertEqual(len(calls),2)
        for call in calls:
            with self.subTest(line=call.lineno):
                captured=io.StringIO()
                with contextlib.redirect_stderr(captured):
                    result=list(eval(compile(ast.Expression(call),str(source),'eval'),{'tqdm':tqdm,'img_paths':list(range(10)),'RANK':0,'self':SimpleNamespace(images=list(range(10)))}))
                self.assertEqual(result,list(range(10)))
                self.assertEqual(captured.getvalue(),'')

    def setUp(self):
        self.recipe = {'disabled_classes': [], 'base_semantics': {
            'prompts': {'1': ['person'], '2': ['car']}, 'score_threshold': .25, 'mask_threshold': .5}}

    def make_job(self, root):
        raw = Path(root) / 'raw'; (raw / 'frames').mkdir(parents=True)
        rows = []
        for i, ts in enumerate((100, 200, 300, 3_000_000_000)):
            np.savez(raw / 'frames' / f'{i:06d}.npz', lidar_timestamp_ns=np.asarray(ts, np.int64))
            rows.append({'frame_id': 100 + i, 'path': f'frames/{i:06d}.npz'})
        (raw / 'index.json').write_text(json.dumps({'rows': rows}))
        return Path(root)

    def fake_infer(self, model, job, rows, indices, prompt, offset, signatures):
        out=[]
        for i in indices:
            mask=np.zeros((1,4,5),bool); mask[0,1:3,2:4]=True
            out.append((mask,np.array([.8],np.float32),np.array([offset+1],np.int64),(4,5),signatures[i]))
        return out

    def test_targeted_refresh_requires_switch_and_tracks_output_changes(self):
        with tempfile.TemporaryDirectory() as d:
            job=self.make_job(d);calls=[]
            def infer(*args):
                calls.append(args[4]);result=self.fake_infer(*args)
                if len(calls)>4:
                    for _,scores,_,_,_ in result:scores[:]=.9
                return result
            with patch.object(av,'_checkpoint',return_value=Path(d)/'model'),patch.object(av,'_load_model',return_value=object()),patch.object(av,'_infer_chunk',side_effect=infer):
                (Path(d)/'model').write_bytes(b'model');av.run(job,self.recipe);before=av.prompt_source_signature(job,'person',self.recipe)
                av.run(job,self.recipe,refresh_prompts=['person']);self.assertEqual(len(calls),4)
                av.run(job,self.recipe,refresh=True,refresh_prompts=['person']);self.assertEqual(calls[4:],['person','person'])
                self.assertNotEqual(av.prompt_source_signature(job,'person',self.recipe),before)

    def test_cache_reuses_unchanged_prompt_and_recomputes_only_invalid_chunk(self):
        with tempfile.TemporaryDirectory() as d:
            job=self.make_job(d); calls=[]; model=object()
            def infer(*args): calls.append((args[4],tuple(args[3]))); return self.fake_infer(*args)
            with patch.object(av,'_checkpoint',return_value=Path(d)/'model'), patch.object(av,'_load_model',return_value=model) as load, patch.object(av,'_infer_chunk',side_effect=infer):
                (Path(d)/'model').write_bytes(b'model')
                first=av.run(job,self.recipe)
                self.assertEqual(len(calls),4)  # two prompts, two temporal chunks
                self.assertEqual(load.call_count,1)
                stable_signature=av.prompt_source_signature(job,'person',self.recipe)
                av.run(job,self.recipe)
                self.assertEqual(len(calls),4)
                self.assertEqual(av.prompt_source_signature(job,'person',self.recipe),stable_signature)
                changed=dict(self.recipe,base_semantics=dict(self.recipe['base_semantics'],prompts={'1':['person edited'],'2':['car']}))
                av.run(job,changed)
                self.assertEqual(len(calls),6)
                self.assertEqual(calls[-2:], [('person edited',(0,1,2)),('person edited',(3,))])
                # Corrupt only car's second chunk; its first chunk and person stay reusable.
                car_root,_=av._actor_root(job,'car')
                cm=json.loads((car_root/'manifest.json').read_text())
                artifact=car_root/cm['chunks'][1]['path']/'frames/000003.npz'
                artifact.write_bytes(b'invalid')
                av.run(job,changed)
                self.assertEqual(len(calls),7)
                self.assertEqual(calls[-1],('car',(3,)))
                self.assertEqual(first['frame_count'],4)

    def test_reader_and_signature(self):
        with tempfile.TemporaryDirectory() as d:
            job=self.make_job(d)
            with patch.object(av,'_checkpoint',return_value=Path(d)/'model'), patch.object(av,'_load_model',return_value=object()), patch.object(av,'_infer_chunk',side_effect=self.fake_infer):
                (Path(d)/'model').write_bytes(b'model')
                av.run(job,self.recipe)
            sig=av.prompt_source_signature(job,'person',self.recipe)
            self.assertTrue(sig)
            result=av.read_prompt(job,'person',0,self.recipe)
            self.assertEqual(result['evidence'].shape,(4,5))
            self.assertAlmostEqual(float(result['evidence'].max()),.8)
            self.assertEqual(result['masks'].shape,(1,4,5))
            self.assertEqual(result['records'][0]['track_id'],result['records'][0]['source_instance_id'])


if __name__ == '__main__': unittest.main()
