import unittest
import numpy as np
from person_track_reference import robust_reference,depth_on_reference_tube,static_reference


class ReferenceTests(unittest.TestCase):
    def test_constant_velocity_remains_constant(self):
        t=np.arange(40)*.1;p=np.c_[t,2*t+1]
        result,_=robust_reference(t,p,np.ones(len(t)));np.testing.assert_allclose(result,p,atol=1e-7)

    def test_mixed_depth_burst_does_not_form_second_point_family(self):
        t=np.arange(40)*.1;p=np.zeros((40,2));p[15:22]=[2,1];weights=np.ones(40);weights[15:22]=.03
        result,_=robust_reference(t,p,weights)
        self.assertLess(np.linalg.norm(result,axis=1).max(),.3)

    def test_bad_prefix_does_not_drag_a_long_moving_track(self):
        t=np.r_[0,.1,.6,1,1.5,np.arange(1.7,5,.2)];true=np.c_[16-t,np.full(len(t),40.)];p=true.copy();p[:2,0]+=8;p[4,0]+=5
        weights=np.full(len(t),.5);weights[4]=.01;result,_=robust_reference(t,p,weights)
        self.assertLess(np.max(np.linalg.norm(result-true,axis=1)),.8)

    def test_stationary_rays_recover_one_centre(self):
        t=np.arange(20)*.1;centre=np.array([10.,5.]);angle=np.linspace(0,1.5,20);origins=np.c_[np.cos(angle)*3,np.sin(angle)*3];steps=centre-origins;reference=np.tile(centre,(20,1))
        p,d=static_reference(t,reference,np.ones(20),origins,steps);np.testing.assert_allclose(p,centre,atol=1e-6)

    def test_moving_track_is_not_collapsed_to_static_cluster(self):
        t=np.arange(20)*.1;p=np.c_[t,np.full(20,5.)];origins=np.zeros((20,2));centre,d=static_reference(t,p,np.ones(20),origins,p)
        self.assertIsNone(centre);self.assertEqual(d['reason'],'moving_track')

    def test_reference_tube_preserves_original_camera_pixel(self):
        T=np.eye(4);E=np.eye(4);E[:3,:3]=[[0,-1,0],[0,0,-1],[1,0,0]];old=np.array([15.,3.,1.]);reference=np.array([5.,1.]);points=np.array([[5.2,1,1],[5.3,1,1],[5.4,1,1]])
        result,ids,status,d=depth_on_reference_tube(old,reference,T,E,points,[1,2,3])
        a=E[:3,:3]@old;b=E[:3,:3]@result;np.testing.assert_allclose(a[:2]/a[2],b[:2]/b[2],atol=1e-12);self.assertLessEqual(np.linalg.norm(result[:2]-reference),.500001)


if __name__=='__main__':unittest.main()
