import copy
import unittest
import numpy as np
from anchor_vehicle_refine import select_anchor,refine_from_anchor
from temporal_refine import vehicle_support_quality,PARAMETERS,wrap_angle
from review_bev_pose import projected_bbox


class AnchorTests(unittest.TestCase):
    def observation(self):
        box=np.array([8.,4.,1.,4.,2.,2.,0.]);E=np.eye(4);E[:3,:3]=[[0,-1,0],[0,0,-1],[1,0,0]]
        K=np.array([[600.,0,960.],[0,600.,540.],[0,0,1.]])
        front=np.c_[np.full(40,6.),np.linspace(3.2,4.8,40),np.ones(40)]
        side=np.c_[np.linspace(6.2,9.8,40),np.full(40,3.),np.ones(40)];points=np.r_[front,side]
        return dict(frame=1,box=box,E=E,K=K,T=np.eye(4),initial_yaw=0.,points=points,bbox=projected_bbox(box,K,E,(1920,1080)),support_quality=vehicle_support_quality(points,box,PARAMETERS))

    def test_bad_frames_cannot_change_selected_anchor(self):
        good=self.observation();poor=copy.deepcopy(good);poor['frame']=2;poor['support_quality']['reliable_anchor']=False;poor['box'][6]=.8
        selected=select_anchor([good]+[poor]*100)
        self.assertIs(selected[0],good)

    def test_anchor_pose_is_never_changed(self):
        anchor=self.observation();result,diag=refine_from_anchor(anchor,anchor)
        np.testing.assert_array_equal(result,anchor['box']);self.assertTrue(diag['anchor_pose_unchanged'])

    def test_other_frames_only_make_bounded_ray_and_heading_corrections(self):
        anchor=self.observation();o=copy.deepcopy(anchor);o['frame']=2;o['box'][:3]*=1.01;o['box'][6]=.06
        result,diag=refine_from_anchor(o,anchor);self.assertIsNotNone(result)
        self.assertLessEqual(abs(diag['camera_depth_adjustment_m']),.25+1e-8)
        self.assertLessEqual(abs(diag['yaw_adjustment_from_anchor_deg']),5.+1e-8)
        np.testing.assert_array_equal(result[3:6],o['box'][3:6])
        before=o['E'][:3,:3]@o['box'][:3];after=o['E'][:3,:3]@result[:3]
        np.testing.assert_allclose(before[:2]/before[2],after[:2]/after[2],atol=1e-10)
        self.assertLessEqual(abs(wrap_angle(result[6]-o['initial_yaw'])),np.pi/2)


if __name__=='__main__':unittest.main()
