import unittest
from unittest.mock import patch, MagicMock
import os
import sys
import ffmpeg

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lib.metadata import get_video_metadata

class TestMetadata(unittest.TestCase):

    @patch("ffmpeg.probe")
    def test_get_video_metadata_success(self, mock_probe):
        mock_meta = {"format": {"duration": "10.5", "size": "1024"}}
        mock_probe.return_value = mock_meta

        res = get_video_metadata("dummy_path.mp4")
        self.assertEqual(res, mock_meta)
        mock_probe.assert_called_once_with("dummy_path.mp4")

    @patch("ffmpeg.probe")
    def test_get_video_metadata_failure(self, mock_probe):
        # Create an ffmpeg.Error mock
        mock_error = ffmpeg.Error(
            cmd="ffprobe dummy_path.mp4",
            stdout=b"",
            stderr=b"Invalid data found when processing input"
        )
        mock_probe.side_effect = mock_error

        with self.assertRaises(RuntimeError) as context:
            get_video_metadata("dummy_path.mp4")
        
        self.assertIn("FFprobe failed to read metadata", str(context.exception))
        self.assertIn("Invalid data found", str(context.exception))

if __name__ == "__main__":
    unittest.main()
