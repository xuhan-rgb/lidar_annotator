import copy
import unittest
import test_anchor_vehicle_refine as fixtures
from odom_local_vehicle import refine_local_odom_tracks
from temporal_refine import PARAMETERS
from prepare_recorded_odom import pose_matrices
import numpy as np


class OdomLocalTests(unittest.TestCase):
    def test_short_track_keeps_valid_single_frame_boxes(self):
        a=fixtures.AnchorTests().observation();a.update(time=0.,status='ignore_short_track',track_id=0)
        b=copy.deepcopy(a);b.update(time=.1,frame=2)
        refine_local_odom_tracks([{'track_id':0,'observations':[a,b]}],PARAMETERS)
        self.assertEqual(a['status'],'accepted_single_frame')
        self.assertEqual(b['status'],'accepted_single_frame')
        self.assertFalse(a['temporal_confirmed'])


    def test_local_anchor_age_never_exceeds_two_seconds(self):
        observations=[]
        for i in range(41):
            o=fixtures.AnchorTests().observation();o.update(frame=i,time=i*.1,status='accepted_temporal',track_id=0)
            observations.append(o)
        refine_local_odom_tracks([{'track_id':0,'observations':observations}],PARAMETERS)
        for o in observations:
            self.assertLessEqual(o['anchor_diagnostics']['anchor_age_s'],2.+1e-6)
            self.assertTrue(o['temporal_confirmed'])

    def test_far_anchor_is_not_used_for_low_density_local_scan(self):
        observations=[]
        for i in range(31):
            o=fixtures.AnchorTests().observation();o.update(frame=i,time=i*.1,status='accepted_temporal',track_id=0)
            if i<30:o['support_quality']['reliable_anchor']=False
            observations.append(o)
        refine_local_odom_tracks([{'track_id':0,'observations':observations}],PARAMETERS)
        self.assertEqual(observations[0]['status'],'accepted_single_frame')
        self.assertEqual(observations[0]['single_frame_diagnostics']['reason'],'no_reliable_local_anchor')

    def test_recorded_pose_is_rebased_without_icp_segment_reset(self):
        poses=np.array([[10,2,0,.1],[10.1,2,0,.1],[10.2,2,0,.1]])
        T=pose_matrices(poses);np.testing.assert_allclose(T[0],np.eye(4),atol=1e-10)
        self.assertAlmostEqual(np.linalg.norm(T[2,:3,3]),.2)


if __name__=='__main__':unittest.main()
