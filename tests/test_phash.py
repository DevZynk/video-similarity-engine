import unittest
from unittest.mock import patch, MagicMock, call
import os
import sys

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lib.phash import (
    get_video_duration,
    extract_sampled_frames,
    generate_phash_fingerprint,
    compare_video_fingerprints_with_timeline
)

class TestPhash(unittest.TestCase):

    @patch("ffmpeg.probe")
    def test_get_video_duration(self, mock_probe):
        mock_probe.return_value = {"format": {"duration": "12.34"}}
        res = get_video_duration("dummy.mp4")
        self.assertEqual(res, 12.34)

        # Check default value if duration is missing
        mock_probe.return_value = {}
        res = get_video_duration("dummy.mp4")
        self.assertEqual(res, 0.0)

    @patch("lib.phash.get_video_duration")
    @patch("ffmpeg.input")
    @patch("glob.glob")
    def test_extract_sampled_frames_success(self, mock_glob, mock_ffmpeg_input, mock_get_duration):
        mock_get_duration.return_value = 5.0
        mock_glob.return_value = [
            "dummy.mp4_frame_00001.jpg",
            "dummy.mp4_frame_00002.jpg",
            "dummy.mp4_frame_00003.jpg",
            "dummy.mp4_frame_00004.jpg"
        ]

        # Mock the fluent interface of ffmpeg: input -> filter -> output -> overwrite_output -> run
        mock_run_obj = MagicMock()
        mock_ffmpeg_input.return_value.filter.return_value.output.return_value.overwrite_output.return_value = mock_run_obj

        res = extract_sampled_frames("dummy.mp4")
        
        # We expect 4 frames
        self.assertEqual(len(res), 4)
        self.assertEqual(res[0], "dummy.mp4_frame_00001.jpg")
        self.assertEqual(res[3], "dummy.mp4_frame_00004.jpg")
        
        # Verify ffmpeg was called
        mock_ffmpeg_input.assert_called_once_with("dummy.mp4", hwaccel="cuda")

    @patch("lib.phash.extract_sampled_frames")
    @patch("PIL.Image.open")
    @patch("imagehash.phash")
    @patch("imagehash.dhash")
    @patch("os.path.exists")
    @patch("os.remove")
    def test_generate_phash_fingerprint(self, mock_remove, mock_exists, mock_dhash, mock_phash, mock_image_open, mock_extract):
        mock_extract.return_value = ["frame0.jpg", "frame1.jpg"]
        mock_exists.return_value = True
        
        # Setup mock return values for imagehash
        mock_phash.side_effect = ["p0", "p1"]
        mock_dhash.side_effect = ["d0", "d1"]
        
        res = generate_phash_fingerprint("dummy.mp4")
        
        self.assertEqual(len(res), 2)
        self.assertEqual(res[0], {"frame_index": 1, "second": 1.0, "phash": "p0", "dhash": "d0"})
        self.assertEqual(res[1], {"frame_index": 2, "second": 2.0, "phash": "p1", "dhash": "d1"})
        
        # Assert cleanups were called
        mock_remove.assert_has_calls([call("frame0.jpg"), call("frame1.jpg")], any_order=True)

    @patch("imagehash.hex_to_hash")
    def test_compare_video_fingerprints_with_timeline(self, mock_hex_to_hash):
        # We need mock hash objects for subtraction
        # imagehash.hex_to_hash returns a hash object that supports subtraction `-` returning an int
        h_sub_p0 = MagicMock()
        h_main_p0 = MagicMock()
        h_sub_d0 = MagicMock()
        h_main_d0 = MagicMock()
        
        # Mock subtraction to return hamming distance
        h_sub_p0.__sub__.return_value = 4   # <= threshold (8)
        h_sub_d0.__sub__.return_value = 10  # > threshold
        
        mock_hex_to_hash.side_effect = [
            h_sub_p0, h_main_p0,  # phash diff = 4
            h_sub_d0, h_main_d0   # dhash diff = 10
        ]
        
        fp1 = [{"phash": "p_sub0", "dhash": "d_sub0"}]
        fp2 = [{"phash": "p_main0", "dhash": "d_main0"}]
        
        res = compare_video_fingerprints_with_timeline(fp1, fp2, threshold=8)
        self.assertTrue(res["is_duplicate"])
        self.assertEqual(res["match_percentage"], 100.0)
        self.assertEqual(res["total_matched_frames"], 1)
        self.assertEqual(len(res["matched_segments"]), 1)
        self.assertEqual(res["matched_segments"][0]["diff_phash"], 4)

    def test_compare_empty_fingerprints(self):
        res = compare_video_fingerprints_with_timeline([], [])
        self.assertFalse(res["is_duplicate"])
        self.assertEqual(res["match_percentage"], 0.0)
        self.assertEqual(res["matched_segments"], [])

if __name__ == "__main__":
    unittest.main()
