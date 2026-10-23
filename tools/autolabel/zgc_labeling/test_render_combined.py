import unittest
from render_combined import clip_bev_edge


class BEVClipTests(unittest.TestCase):
    def test_vehicle_edge_does_not_draw_outside_bev_panel(self):
        self.assertEqual(clip_bev_edge((1500,50),(1500,110)),((1500.,70.),(1500.,110.)))
        self.assertIsNone(clip_bev_edge((1500,20),(1600,50)))


if __name__=='__main__':unittest.main()
