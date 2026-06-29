import unittest
import sys
from unittest.mock import MagicMock, patch

# Intercept scenedetect and open_clip modules
mock_scenedetect = MagicMock()
sys.modules['scenedetect'] = mock_scenedetect

mock_open_clip = MagicMock()
mock_open_clip.create_model_and_transforms.return_value = (MagicMock(), None, MagicMock())
sys.modules['open_clip'] = mock_open_clip

# Ensure project root is in path
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.compare import compare_video

class TestCompareOrchestrator(unittest.TestCase):

    @patch("core.compare.config")
    @patch("core.compare.blake3_compare_fingerprint")
    def test_compare_blake3_duplicate(self, mock_blake_compare, mock_config):
        mock_config.__getitem__.side_effect = lambda k: {
            "features": {
                "blake3": {"enabled": True}
            }
        }.get(k, {})
        mock_blake_compare.return_value = True

        v1 = {"blake": "hash1"}
        v2 = {"blake": "hash1"}
        res = compare_video(v1, v2)
        
        self.assertEqual(res["status"], "duplicate")
        self.assertEqual(res["match_type"], "blake3")
        self.assertEqual(res["match_percentage"], 100.0)

    @patch("core.compare.config")
    @patch("core.compare.blake3_compare_fingerprint")
    @patch("core.compare.compare_video_fingerprints_with_timeline")
    def test_compare_phash_duplicate(self, mock_phash_compare, mock_blake_compare, mock_config):
        mock_config.__getitem__.side_effect = lambda k: {
            "features": {
                "blake3": {"enabled": True},
                "phash": {"enabled": True, "duplicate_threshold_pct": 90.0}
            }
        }.get(k, {})
        
        # Blake3 doesn't match, phash matches above threshold
        mock_blake_compare.return_value = False
        mock_phash_compare.return_value = {
            "match_percentage": 95.0,
            "matched_segments": [{"dummy": "segment"}]
        }

        res = compare_video({"phash": []}, {"phash": []})
        self.assertEqual(res["status"], "duplicate")
        self.assertEqual(res["match_type"], "phash")
        self.assertEqual(res["match_percentage"], 95.0)

    @patch("core.compare.config")
    @patch("core.compare.blake3_compare_fingerprint")
    @patch("core.compare.compare_video_fingerprints_with_timeline")
    @patch("core.compare.compare_ai_fingerprints")
    def test_compare_ai_duplicate(self, mock_ai_compare, mock_phash_compare, mock_blake_compare, mock_config):
        mock_config.__getitem__.side_effect = lambda k: {
            "features": {
                "blake3": {"enabled": True},
                "phash": {"enabled": True, "duplicate_threshold_pct": 90.0},
                "ai_embedding": {"enabled": True, "duplicate_threshold_pct": 80.0}
            }
        }.get(k, {})
        
        mock_blake_compare.return_value = False
        mock_phash_compare.return_value = {"match_percentage": 10.0}
        mock_ai_compare.return_value = {
            "match_percentage": 85.0,
            "matched_segments": ["segment_ai"]
        }

        res = compare_video({}, {})
        self.assertEqual(res["status"], "duplicate")
        self.assertEqual(res["match_type"], "ai_embedding")
        self.assertEqual(res["match_percentage"], 85.0)

    @patch("core.compare.config")
    @patch("core.compare.blake3_compare_fingerprint")
    @patch("core.compare.compare_video_fingerprints_with_timeline")
    @patch("core.compare.compare_ai_fingerprints")
    @patch("core.compare.compare_audio_fingerprints")
    @patch("core.compare.compare_motion_fingerprints")
    @patch("core.compare.compare_scene_structures")
    def test_compare_not_duplicate(
        self, mock_scene, mock_motion, mock_audio, mock_ai, mock_phash, mock_blake, mock_config
    ):
        # Enable all features
        mock_config.__getitem__.side_effect = lambda k: {
            "features": {
                "blake3": {"enabled": True},
                "phash": {"enabled": True, "duplicate_threshold_pct": 90.0},
                "ai_embedding": {"enabled": True, "duplicate_threshold_pct": 90.0},
                "audio": {"enabled": True, "duplicate_threshold_pct": 90.0},
                "motion": {"enabled": True, "duplicate_threshold_pct": 90.0},
                "scene": {"enabled": True, "duplicate_threshold_pct": 90.0}
            }
        }.get(k, {})

        # Set all matches below threshold (90.0)
        mock_blake.return_value = False
        mock_phash.return_value = {"match_percentage": 15.0}
        mock_ai.return_value = {"match_percentage": 25.0}
        mock_audio.return_value = {"match_percentage": 45.0, "matched_segments": ["audio_seg"]}
        mock_motion.return_value = {"match_percentage": 10.0}
        mock_scene.return_value = {"match_percentage": 5.0}

        res = compare_video({}, {})
        
        self.assertEqual(res["status"], "not_duplicate")
        # Best match is audio (45.0)
        self.assertEqual(res["best_match_method"], "audio")
        self.assertEqual(res["best_match_percentage"], 45.0)
        self.assertEqual(res["matched_segments"], ["audio_seg"])

if __name__ == "__main__":
    unittest.main()
