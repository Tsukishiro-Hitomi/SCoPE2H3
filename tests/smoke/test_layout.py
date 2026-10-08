import unittest
from pathlib import Path

from scope2h3.inference import aligned_frames, validate_ref2va_model_dir


class LayoutSmokeTests(unittest.TestCase):
    def test_h3_duration_alignment(self):
        self.assertEqual(aligned_frames(20), 481)
        self.assertEqual(aligned_frames(1 / 24), 5)

    def test_missing_model_directory_fails_before_gpu_initialisation(self):
        with self.assertRaises(FileNotFoundError):
            validate_ref2va_model_dir(Path("does-not-exist"))

    def test_controlled_runtime_and_owned_package_exist(self):
        root = Path(__file__).resolve().parents[2]
        self.assertTrue((root / "src/diffsynth/pipelines/minimax_h3_audio_video.py").is_file())
        self.assertTrue((root / "src/scope2h3/adapter.py").is_file())
