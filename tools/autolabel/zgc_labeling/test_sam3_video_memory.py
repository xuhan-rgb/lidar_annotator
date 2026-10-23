import unittest
from types import SimpleNamespace
from run_sam3_full_video import offload_tracker_state,prune_exported_output_cache,remap_object_ids,split_temporal_chunks


class MemoryTests(unittest.TestCase):
    def test_every_new_tracker_state_uses_cpu_history(self):
        model=SimpleNamespace(tracker=SimpleNamespace(init_state=lambda **kw:kw))
        offload_tracker_state(model)
        self.assertTrue(model.tracker.init_state(video_height=1080)['offload_state_to_cpu'])

    def test_pruning_keeps_recent_output_cache_and_tracker_memory(self):
        memory=object();state={'cached_frame_outputs':{0:0,68:1,69:2,100:3},'tracker_inference_states':memory}
        prune_exported_output_cache(state,100)
        self.assertEqual(set(state['cached_frame_outputs']),{68,69,100});self.assertIs(state['tracker_inference_states'],memory)

    def test_temporal_gaps_and_nonmonotonic_time_start_new_chunks(self):
        self.assertEqual(split_temporal_chunks([10,20,2_000_000_021,2_000_000_022,2_000_000_020]),
                         [[0,1],[2,3],[4]])

    def test_chunk_ids_are_offset_without_changing_local_order(self):
        ids=remap_object_ids([0,2,7],8)
        self.assertEqual(ids.tolist(),[8,10,15])


if __name__=='__main__':unittest.main()
