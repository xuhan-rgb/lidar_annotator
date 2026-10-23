import unittest
from deduplicate_vehicle_map import bev_iou,deduplicate


class DedupTests(unittest.TestCase):
    def item(self,tid,x,score,yaw=.3):
        return dict(track_id=tid,candidate_box_map=[x,0,1,4,2,2,yaw],anchor_quality=dict(score=score))

    def test_duplicate_retains_better_anchor_without_averaging(self):
        poor=self.item(2,.1,1);good=self.item(7,0,3)
        kept,removed=deduplicate([poor,good]);self.assertEqual(kept,[good]);self.assertEqual(removed[0]['kept_track_id'],7)

    def test_adjacent_vehicles_are_retained(self):
        kept,removed=deduplicate([self.item(0,0,3),self.item(1,5,2)])
        self.assertEqual(len(kept),2);self.assertFalse(removed)

    def test_overlap_uses_rotated_rectangles(self):
        a=self.item(0,0,1,0)['candidate_box_map'];b=self.item(1,0,1,1.5707963267948966)['candidate_box_map']
        self.assertAlmostEqual(bev_iou(a,b),1/3)


if __name__=='__main__':unittest.main()
