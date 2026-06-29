import unittest
import sys
from unittest.mock import MagicMock, patch

# Ensure project root is in path
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Intercept scenedetect to prevent import error
if 'scenedetect' not in sys.modules:
    sys.modules['scenedetect'] = MagicMock()

from lib.scene_detector import detect_video_scenes, compare_scene_structures

class TestSceneDetector(unittest.TestCase):

    @patch("lib.scene_detector.open_video")
    @patch("lib.scene_detector.SceneManager")
    @patch("lib.scene_detector.ContentDetector")
    def test_detect_video_scenes_success(self, mock_content_detector_cls, mock_scene_manager_cls, mock_open_video):
        # Setup mocks
        mock_video = MagicMock()
        mock_open_video.return_value = mock_video
        
        mock_scene_manager = MagicMock()
        mock_scene_manager_cls.return_value = mock_scene_manager
        
        # Create mock scene timecodes
        mock_start = MagicMock()
        mock_start.seconds = 0.0
        mock_end = MagicMock()
        mock_end.seconds = 5.4
        
        mock_scene = (mock_start, mock_end)
        mock_scene_manager.get_scene_list.return_value = [mock_scene]
        
        res = detect_video_scenes("dummy_video.mp4", threshold=27.0)
        
        self.assertEqual(res["video_path"], "dummy_video.mp4")
        self.assertEqual(res["total_scenes"], 1)
        self.assertEqual(res["scenes"][0]["scene_index"], 1)
        self.assertEqual(res["scenes"][0]["start_second"], 0.0)
        self.assertEqual(res["scenes"][0]["end_second"], 5.4)
        self.assertEqual(res["scenes"][0]["duration_seconds"], 5.4)
        
        # Assertions
        mock_open_video.assert_called_once_with("dummy_video.mp4")
        mock_scene_manager_cls.assert_called_once()
        mock_content_detector_cls.assert_called_once_with(threshold=27.0)
        mock_scene_manager.add_detector.assert_called_once()
        mock_scene_manager.detect_scenes.assert_called_once_with(mock_video)

    def test_compare_scene_structures_match(self):
        scenes1 = [
            {"scene_index": 1, "start_second": 0.0, "end_second": 5.0, "duration_seconds": 5.0},
            {"scene_index": 2, "start_second": 5.0, "end_second": 12.0, "duration_seconds": 7.0}
        ]
        
        scenes2 = [
            {"scene_index": 1, "start_second": 0.2, "end_second": 5.1, "duration_seconds": 4.9},
            {"scene_index": 2, "start_second": 5.1, "end_second": 12.2, "duration_seconds": 7.1}
        ]
        
        res = compare_scene_structures(scenes1, scenes2, time_tolerance=1.0)
        
        self.assertTrue(res["is_structure_duplicate"])
        self.assertEqual(res["match_percentage"], 100.0)
        self.assertEqual(res["total_matched_scenes"], 2)

    def test_compare_scene_structures_no_match(self):
        scenes1 = [
            {"scene_index": 1, "start_second": 0.0, "end_second": 5.0, "duration_seconds": 5.0}
        ]
        scenes2 = [
            {"scene_index": 1, "start_second": 10.0, "end_second": 25.0, "duration_seconds": 15.0}
        ]
        
        res = compare_scene_structures(scenes1, scenes2, time_tolerance=1.0)
        self.assertFalse(res["is_structure_duplicate"])
        self.assertEqual(res["match_percentage"], 0.0)

    def test_compare_scene_structures_empty(self):
        res = compare_scene_structures([], [])
        self.assertFalse(res["is_structure_duplicate"])
        self.assertEqual(res["match_percentage"], 0.0)

if __name__ == "__main__":
    unittest.main()
