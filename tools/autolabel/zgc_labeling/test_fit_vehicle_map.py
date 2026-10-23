import unittest
import numpy as np
from fit_vehicle_map import edge_distances,fit_history


class HistoryFitTests(unittest.TestCase):
    def test_edges_use_each_frames_sensor_origin(self):
        p=np.array([[3.,0.,0.],[7.,0.,0.]])
        origins=np.array([[0.,0.,0.],[10.,0.,0.]])
        np.testing.assert_allclose(edge_distances(p,[5,0,0],0,np.array([4,2,2]),origins),0.)

    def test_history_recovers_depth_preserving_ray_size_and_yaw_bound(self):
        points=np.r_[np.c_[np.full(80,6.),np.linspace(-1,1,80),np.ones(80)],np.c_[np.linspace(6,10,80),np.full(80,-1.),np.ones(80)]]
        rng=np.random.default_rng(3);points=np.r_[points, rng.uniform([3,-4,0],[13,4,2],(20,3))]
        box=np.array([8.7,0,1,4,2,2,.15]);origin=np.array([0,0,1.]);step=np.array([1.,0,0])
        new,diag=fit_history(points,np.tile([0,-4,0],(len(points),1)),np.zeros(len(points)),box,origin,step,8.7,0.)
        self.assertTrue(diag['converged']);self.assertLess(abs(new[0]-8),.15)
        np.testing.assert_array_equal(new[3:6],box[3:6]);self.assertAlmostEqual(new[1],0.)
        self.assertLessEqual(abs(new[6]),np.pi/2);self.assertLess(diag['objective_after'],diag['objective_before'])


if __name__=='__main__':unittest.main()
