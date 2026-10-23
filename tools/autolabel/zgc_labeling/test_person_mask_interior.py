import unittest
import numpy as np
from person_mask_interior import select_interior_half


class InteriorTests(unittest.TestCase):
    def test_selects_interior_half_instead_of_boundary(self):
        mask=np.zeros((30,30),bool);mask[5:25,5:25]=True
        uv=np.array([[5,5],[6,15],[15,15],[14,14]])
        selected,scores,diag=select_interior_half(mask,uv,np.arange(4))
        self.assertEqual(set(selected),{2,3});self.assertEqual(diag['selected_points'],2)

    def test_excludes_outside_mask_and_keeps_ceil_half(self):
        mask=np.ones((10,10),bool);mask[3,3]=False
        uv=np.array([[3,3],[20,20],[1,1],[5,5],[6,6]])
        ids,_,diag=select_interior_half(mask,uv,np.arange(5))
        self.assertEqual(diag['valid_points'],3);self.assertEqual(len(ids),2)

    def test_holes_do_not_become_outer_boundary(self):
        mask=np.zeros((30,30),bool);mask[2:28,2:28]=True;mask[14,14]=False
        uv=np.array([[15,14],[3,3]])
        selected,_,_=select_interior_half(mask,uv,[0,1]);self.assertEqual(selected.tolist(),[0])

    def test_empty_support_returns_empty(self):
        ids,_,_=select_interior_half(np.zeros((5,5),bool),np.empty((0,2)),[])
        self.assertEqual(len(ids),0)


if __name__=='__main__':unittest.main()
