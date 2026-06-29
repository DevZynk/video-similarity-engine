import unittest
import sys
import tempfile
import os
from unittest.mock import MagicMock, patch

# Intercept scenedetect and open_clip modules
mock_scenedetect = MagicMock()
sys.modules['scenedetect'] = mock_scenedetect

mock_open_clip = MagicMock()
mock_open_clip.create_model_and_transforms.return_value = (MagicMock(), None, MagicMock())
sys.modules['open_clip'] = mock_open_clip

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import main and FastAPI test client
from fastapi.testclient import TestClient
import main
from main import app

class TestAPI(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory for fingerprints database
        self.test_dir = tempfile.TemporaryDirectory()
        self.original_db_dir = main.DB_DIR
        main.DB_DIR = self.test_dir.name
        self.client = TestClient(app)

    def tearDown(self):
        main.DB_DIR = self.original_db_dir
        self.test_dir.cleanup()

    @patch("main.video_fingerprint_per_second")
    @patch("os.path.exists")
    def test_fingerprint_video_success(self, mock_exists, mock_fingerprint):
        mock_exists.return_value = True # File exists check in main.py
        mock_fingerprint.return_value = {
            "status": "success",
            "blake": "abc123blake",
            "metadata": {"duration": 15.0}
        }

        response = self.client.post("/fingerprint?path=dummy_video.mp4")
        
        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertEqual(json_data["status"], "success")
        self.assertEqual(json_data["blake3_hash"], "abc123blake")
        
        # Verify JSON file was written
        saved_file = os.path.join(main.DB_DIR, "abc123blake.json")
        self.assertTrue(os.path.exists(saved_file))

    @patch("os.path.exists")
    def test_fingerprint_video_file_not_found(self, mock_exists):
        mock_exists.return_value = False
        
        response = self.client.post("/fingerprint?path=non_existent.mp4")
        self.assertEqual(response.status_code, 404)
        self.assertIn("Video file not found", response.json()["detail"])

    def test_get_fingerprint_not_found(self):
        response = self.client.get("/fingerprint/non_existent_hash")
        self.assertEqual(response.status_code, 404)

    def test_get_fingerprint_success(self):
        # Write dummy fingerprint JSON to DB_DIR
        import json
        dummy_data = {"blake": "mockedhash", "status": "success"}
        with open(os.path.join(main.DB_DIR, "mockedhash.json"), "w") as f:
            json.dump(dummy_data, f)
            
        response = self.client.get("/fingerprint/mockedhash")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), dummy_data)

    @patch("main.compare_video")
    def test_compare_videos_by_hash(self, mock_compare_video):
        # Write dummy fingerprints
        import json
        with open(os.path.join(main.DB_DIR, "hash1.json"), "w") as f:
            json.dump({"blake": "hash1", "status": "success"}, f)
        with open(os.path.join(main.DB_DIR, "hash2.json"), "w") as f:
            json.dump({"blake": "hash2", "status": "success"}, f)

        mock_compare_video.return_value = {"status": "duplicate", "match_type": "phash"}

        response = self.client.post("/compare?video_source=hash1&video_target=hash2")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "duplicate")

if __name__ == "__main__":
    unittest.main()
