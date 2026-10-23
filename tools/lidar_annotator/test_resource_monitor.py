"""Pure parser and CPU-delta checks for ResourceMonitor."""
import unittest

from .resource_monitor import cpu_utilization, parse_nvidia_smi_output


class ResourceMonitorTests(unittest.TestCase):
    def test_parse_multiple_gpu_rows(self):
        rows = parse_nvidia_smi_output(
            "NVIDIA RTX A4000, 37, 4096, 16376\n"
            "NVIDIA RTX 4090, 82, 12000, 24564\n"
        )
        self.assertEqual(rows, [
            ("NVIDIA RTX A4000", 37, 4096, 16376),
            ("NVIDIA RTX 4090", 82, 12000, 24564),
        ])

    def test_parser_skips_invalid_or_unavailable_rows(self):
        self.assertEqual(parse_nvidia_smi_output(
            "\nGPU 0, N/A, N/A, N/A\n"
            "GPU 1, 101, 2, 4\n"
            "GPU 2, 20, 2, 0\n"
            "GPU 3, 20, 2, 4"
        ), [("GPU 3", 20, 2, 4)])

    def test_cpu_interval_delta(self):
        self.assertIsNone(cpu_utilization(None, (10, 0, 0, 5, 0)))
        self.assertAlmostEqual(cpu_utilization((10, 0, 0, 5, 0), (20, 0, 0, 7, 1)), 100 * 10 / 13)
        self.assertIsNone(cpu_utilization((2, 2), (3, 3)))
        self.assertIsNone(cpu_utilization((10, 0, 0, 5, 0), (10, 0, 0, 5, 0)))


if __name__ == "__main__":
    unittest.main()
