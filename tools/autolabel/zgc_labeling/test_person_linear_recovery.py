import unittest
import numpy as np
from person_linear_recovery import find_outliers,recover_from_mask,camera_ray_recovery


class LinearRecoveryTests(unittest.TestCase):
    def test_two_consecutive_background_positions_do_not_remove_good_neighbours(self):
        frames=np.arange(12);times=frames*.1;xy=np.c_[times,np.ones(12)];xy[4:6,1]=15
        bad,pred=find_outliers(frames,times,xy)
        self.assertEqual(np.flatnonzero(bad).tolist(),[4,5])
        for i in [4,5]:np.testing.assert_allclose(pred[i]['xy'],[times[i],1],atol=1e-5)

    def test_one_noisy_neighbour_does_not_hide_large_central_jump(self):
        f=np.arange(81,86);xy=np.array([[10.38,-10],[10.37,-10.08],[10.09,-23.30],[10.29,-10],[10.3,-11.69]])
        bad,pred=find_outliers(f,np.arange(5)*.1,xy)
        self.assertTrue(bad[2]);self.assertLessEqual(np.linalg.norm(pred[2]['velocity']),3.+1e-6)

    def test_terminal_jump_is_detected_from_reliable_history(self):
        f=np.arange(10);xy=np.c_[f*.1,np.ones(10)];xy[-1]=[15,20]
        bad,pred=find_outliers(f,f*.1,xy);self.assertTrue(bad[-1]);self.assertFalse(bad[:-1].any())

    def test_three_frame_bad_run_is_detected_with_bounded_wider_window(self):
        f=np.arange(15);xy=np.c_[f*.1,np.ones(15)];xy[6:9,1]=15
        bad,pred=find_outliers(f,f*.1,xy);self.assertTrue(bad[6:9].all());self.assertFalse(bad[:6].any());self.assertFalse(bad[9:].any())

    def test_metric_gap_can_be_checked_only_with_continuous_sam_presence(self):
        f=np.array([178,184,185,186,187,188]);t=(f-178)*.1;xy=np.tile([30.,-20.],(6,1));xy[0]=[20,-16]
        bad,_=find_outliers(f,t,xy);self.assertFalse(bad[0])
        bad,_=find_outliers(f,t,xy,set(range(178,189)));self.assertTrue(bad[0])
        bad,_=find_outliers(f,t,xy,{178,184,185,186,187,188});self.assertFalse(bad[0])

    def test_constant_velocity_is_not_an_outlier(self):
        f=np.arange(15);xy=np.c_[f*.15,f*.05]
        bad,_=find_outliers(f,f*.1,xy);self.assertFalse(bad.any())

    def test_current_mask_points_replace_prediction_when_supported(self):
        pts=np.array([[1,2,0],[1.1,2,0],[1,2.1,0],[20,20,0]])
        p,ids,status=recover_from_mask(pts,np.arange(4),[1,2]);self.assertEqual(ids.tolist(),[0,1,2]);np.testing.assert_allclose(p,[1,2,0]);self.assertEqual(status,'linear_guided_mask_lidar')

    def test_missing_support_remains_explicit_interpolation(self):
        p,ids,status=recover_from_mask(np.array([[20,20,0]]),[1],[1,2]);self.assertIsNone(p);self.assertEqual(len(ids),0);self.assertEqual(status,'linear_interpolation_only')

    def test_camera_ray_preserves_pixel_with_camera_lidar_translation(self):
        T=np.eye(4);E=np.eye(4);E[:3,:3]=[[0,-1,0],[0,0,-1],[1,0,0]];E[:3,3]=[.1,-.2,.3]
        original=np.array([10.,2.,1.]);cam=E[:3,:3]@original+E[:3,3];inverse=np.linalg.inv(E);step=inverse[:3,:3]@(cam/cam[2]);origin=inverse[:3,3];target=origin+6*step
        points=np.array([target+[0,.03,0],target,target+[0,-.03,0]])
        point,ids,status,diag=camera_ray_recovery(original,target[:2],points,[1,2,3],T,E)
        result=E[:3,:3]@point+E[:3,3]
        np.testing.assert_allclose(result[:2]/result[2],cam[:2]/cam[2],atol=1e-12);self.assertEqual(status,'linear_guided_mask_lidar');self.assertEqual(len(ids),3)

    def test_conflicting_camera_ray_is_ignored(self):
        T=np.eye(4);E=np.eye(4);E[:3,:3]=[[0,-1,0],[0,0,-1],[1,0,0]]
        point,ids,status,diag=camera_ray_recovery([10,0,1],[5,10],np.empty((0,3)),[],T,E)
        self.assertIsNone(point);self.assertEqual(status,'ignored_camera_ray_conflict')

    def test_long_gap_is_not_bridged(self):
        f=np.arange(6);times=np.array([0,.1,.2,3,3.1,3.2]);xy=np.zeros((6,2));xy[2]=[20,20]
        bad,_=find_outliers(f,times,xy);self.assertFalse(bad.any())


if __name__=='__main__':unittest.main()
