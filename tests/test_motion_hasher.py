import unittest
from unittest.mock import patch, MagicMock
import os
import sys
import numpy as np

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lib.motion_hasher import generate_video_motion_fingerprint, compare_motion_fingerprints

class TestMotionHasher(unittest.TestCase):

    @patch("cv2.VideoCapture")
    @patch("cv2.cvtColor")
    @patch("cv2.resize")
    @patch("cv2.calcOpticalFlowFarneback")
    @patch("cv2.cartToPolar")
    def test_generate_video_motion_fingerprint_success(
        self, mock_cartToPolar, mock_calcFlow, mock_resize, mock_cvtColor, mock_VideoCapture
    ):
        # Mock VideoCapture
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 25.0 # FPS = 25
        
        # cap.read() should return True for the first frame, and then True and False
        # loop runs: first read outside loop, then set pos frame, then read inside loop, then break
        mock_cap.read.side_effect = [
            (True, "frame1"), # outside loop
            (True, "frame2"), # first iteration
            (False, None)     # second iteration (break)
        ]
        mock_VideoCapture.return_value = mock_cap

        # Mock opencv processing functions
        mock_cvtColor.return_value = "gray_frame"
        mock_resize.return_value = np.zeros((256, 256), dtype=np.uint8)
        
        # calcOpticalFlowFarneback returns flow field (x,y)
        mock_calcFlow.return_value = np.zeros((256, 256, 2), dtype=np.float32)
        
        # cartToPolar returns magnitude and angle
        mock_mag = np.ones((256, 256), dtype=np.float32) * 1.5
        mock_ang = np.ones((256, 256), dtype=np.float32) * 0.5
        mock_cartToPolar.return_value = (mock_mag, mock_ang)

        res = generate_video_motion_fingerprint("dummy_video.mp4")
        
        self.assertEqual(res["video_path"], "dummy_video.mp4")
        self.assertEqual(len(res["motion_fingerprints"]), 1)
        self.assertEqual(res["motion_fingerprints"][0]["frame_index"], 1)
        self.assertEqual(res["motion_fingerprints"][0]["second"], 1.0)
        self.assertEqual(res["motion_fingerprints"][0]["magnitude"], 1.5)
        self.assertEqual(res["motion_fingerprints"][0]["angle"], 0.5)
        
        # Assert calls
        mock_cap.release.assert_called_once()
        self.assertEqual(mock_cap.set.call_count, 2)

    @patch("cv2.VideoCapture")
    def test_generate_video_motion_fingerprint_not_opened(self, mock_VideoCapture):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_VideoCapture.return_value = mock_cap

        res = generate_video_motion_fingerprint("dummy_video.mp4")
        self.assertEqual(res["motion_fingerprints"], [])

    @patch("cv2.VideoCapture")
    def test_generate_video_motion_fingerprint_no_frames(self, mock_VideoCapture):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 25.0
        mock_cap.read.return_value = (False, None)
        mock_VideoCapture.return_value = mock_cap

        res = generate_video_motion_fingerprint("dummy_video.mp4")
        self.assertEqual(res["motion_fingerprints"], [])
        mock_cap.release.assert_called_once()

    def test_compare_motion_fingerprints_match(self):
        fp1 = [{"magnitude": 1.5, "angle": 0.5}, {"magnitude": 2.0, "angle": 1.0}]
        # Identical or very similar
        fp2 = [{"magnitude": 1.55, "angle": 0.52}, {"magnitude": 1.98, "angle": 0.98}]
        
        # MSE = ((diff_mag**2) + (diff_ang**2)) / 2
        # For item 1: ((0.05**2) + (0.02**2)) / 2 = (0.0025 + 0.0004)/2 = 0.00145 <= threshold (1.5)
        # For item 2: ((0.02**2) + (0.02**2)) / 2 = (0.0004 + 0.0004)/2 = 0.0004 <= threshold (1.5)
        res = compare_motion_fingerprints(fp1, fp2, max_mse_threshold=1.5)
        
        self.assertTrue(res["is_motion_duplicate"])
        self.assertEqual(res["match_percentage"], 100.0)
        self.assertEqual(res["total_matched_frames"], 2)

    def test_compare_motion_fingerprints_no_match(self):
        fp1 = [{"magnitude": 1.0, "angle": 0.0}]
        fp2 = [{"magnitude": 10.0, "angle": 5.0}]
        # MSE = ((9**2) + (5**2))/2 = (81 + 25)/2 = 53.0 > threshold (1.5)
        res = compare_motion_fingerprints(fp1, fp2, max_mse_threshold=1.5)
        
        self.assertFalse(res["is_motion_duplicate"])
        self.assertEqual(res["match_percentage"], 0.0)
        self.assertEqual(res["total_matched_frames"], 0)

    def test_compare_motion_fingerprints_empty(self):
        res = compare_motion_fingerprints([], [])
        self.assertFalse(res["is_motion_duplicate"])
        self.assertEqual(res["match_percentage"], 0.0)

if __name__ == "__main__":
    unittest.main()
