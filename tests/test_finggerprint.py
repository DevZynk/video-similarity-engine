import unittest
from unittest.mock import patch, MagicMock
import os
import sys

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

from core.finggerprint import video_fingerprint_per_second

class TestFingerprintOrchestrator(unittest.TestCase):

    @patch("core.finggerprint.temp_file")
    @patch("core.finggerprint.delete_temp_file")
    @patch("core.finggerprint.blake3_file_fingerprint")
    @patch("core.finggerprint.get_video_metadata")
    @patch("core.finggerprint.generate_phash_fingerprint")
    @patch("core.finggerprint.generate_audio_fingerprint")
    @patch("core.finggerprint.generate_video_ai_embeddings")
    @patch("core.finggerprint.generate_video_motion_fingerprint")
    @patch("core.finggerprint.detect_video_scenes")
    @patch("core.finggerprint.config")
    def test_video_fingerprint_all_enabled(
        self, mock_config, mock_scene, mock_motion, mock_ai, mock_audio, mock_phash,
        mock_meta, mock_blake, mock_delete_temp, mock_temp_file
    ):
        # Enable all features in mock config
        mock_config.__getitem__.side_effect = lambda k: {
            "features": {
                "blake3": {"enabled": True},
                "phash": {"enabled": True},
                "audio": {"enabled": True},
                "ai_embedding": {"enabled": True},
                "motion": {"enabled": True},
                "scene": {"enabled": True}
            }
        }.get(k, {})

        # Setup mock returns
        mock_temp_file.return_value = "dummy_temp_path.mp4"
        mock_blake.return_value = "blake_hash_123"
        mock_meta.return_value = {"duration": 10.0}
        mock_phash.return_value = [{"phash": "a", "dhash": "b"}]
        mock_audio.return_value = {"audio_subprints": [1, 2]}
        mock_ai.return_value = {"ai_embeddings": [[0.1, 0.2]]}
        mock_motion.return_value = {"motion_fingerprints": [{"magnitude": 1.0, "angle": 0.5}]}
        mock_scene.return_value = {"total_scenes": 1, "scenes": []}

        res = video_fingerprint_per_second("dummy_data", "file")
        
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["blake"], "blake_hash_123")
        self.assertEqual(res["metadata"], {"duration": 10.0})
        self.assertEqual(res["phash"], [{"phash": "a", "dhash": "b"}])
        self.assertEqual(res["audio"]["audio_subprints"], [1, 2])
        self.assertEqual(res["ai"]["ai_embeddings"], [[0.1, 0.2]])
        self.assertEqual(res["motion"]["motion_fingerprints"], [{"magnitude": 1.0, "angle": 0.5}])
        self.assertEqual(res["scene"]["total_scenes"], 1)

        # Verify temp file cleanup
        mock_temp_file.assert_called_once_with("file", data="dummy_data")
        mock_delete_temp.assert_called_once_with("dummy_temp_path.mp4")

    @patch("core.finggerprint.temp_file")
    @patch("core.finggerprint.delete_temp_file")
    @patch("core.finggerprint.get_video_metadata")
    @patch("core.finggerprint.config")
    def test_video_fingerprint_all_disabled(
        self, mock_config, mock_meta, mock_delete_temp, mock_temp_file
    ):
        # Disable all optional features in mock config
        mock_config.__getitem__.side_effect = lambda k: {
            "features": {
                "blake3": {"enabled": False},
                "phash": {"enabled": False},
                "audio": {"enabled": False},
                "ai_embedding": {"enabled": False},
                "motion": {"enabled": False},
                "scene": {"enabled": False}
            }
        }.get(k, {})

        mock_temp_file.return_value = "dummy_temp_path.mp4"
        mock_meta.return_value = {"duration": 10.0}

        res = video_fingerprint_per_second("dummy_data", "file")
        
        self.assertEqual(res["status"], "success")
        self.assertIsNone(res["blake"])
        self.assertEqual(res["phash"], [])
        self.assertEqual(res["audio"]["audio_subprints"], [])
        self.assertEqual(res["ai"]["ai_embeddings"], [])
        self.assertEqual(res["motion"]["motion_fingerprints"], [])
        self.assertEqual(res["scene"]["total_scenes"], 0)

    @patch("core.finggerprint.temp_file")
    @patch("core.finggerprint.delete_temp_file")
    def test_video_fingerprint_exception(self, mock_delete_temp, mock_temp_file):
        mock_temp_file.side_effect = Exception("Failed to download file")
        
        res = video_fingerprint_per_second("dummy_data", "url")
        self.assertEqual(res["status"], "error")
        self.assertEqual(res["message"], "Failed to download file")
        
        # Verify temp file deletion wasn't called since file_path was None
        mock_delete_temp.assert_not_called()

if __name__ == "__main__":
    unittest.main()
