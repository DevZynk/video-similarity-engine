import unittest
import sys
from unittest.mock import MagicMock, patch, call

# Mock open_clip BEFORE importing lib.ai_embedding to avoid loading heavy CLIP model
mock_open_clip = MagicMock()
mock_model = MagicMock()
mock_preprocess = MagicMock()
mock_open_clip.create_model_and_transforms.return_value = (mock_model, None, mock_preprocess)

sys.modules['open_clip'] = mock_open_clip

# Now import the module under test
import os
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lib.ai_embedding import (
    get_video_duration,
    extract_sampled_frames_cuda,
    generate_video_ai_embeddings,
    compare_ai_fingerprints,
    device
)

class TestAiEmbedding(unittest.TestCase):

    def setUp(self):
        mock_model.reset_mock()
        mock_preprocess.reset_mock()

    @patch("ffmpeg.probe")
    def test_get_video_duration(self, mock_probe):
        mock_probe.return_value = {"format": {"duration": "60.0"}}
        res = get_video_duration("dummy.mp4")
        self.assertEqual(res, 60.0)

    @patch("lib.ai_embedding.get_video_duration")
    @patch("ffmpeg.input")
    @patch("glob.glob")
    def test_extract_sampled_frames_cuda_success(self, mock_glob, mock_ffmpeg_input, mock_get_duration):
        mock_get_duration.return_value = 3.0
        mock_glob.return_value = ["dummy.mp4_ai_frame_00001.jpg", "dummy.mp4_ai_frame_00002.jpg"]

        mock_run_obj = MagicMock()
        mock_ffmpeg_input.return_value.filter.return_value.output.return_value.overwrite_output.return_value = mock_run_obj

        res = extract_sampled_frames_cuda("dummy.mp4")
        self.assertEqual(len(res), 2)
        self.assertEqual(res[0], "dummy.mp4_ai_frame_00001.jpg")
        
        mock_ffmpeg_input.assert_called_once_with("dummy.mp4", hwaccel="cuda")

    @patch("lib.ai_embedding.extract_sampled_frames_cuda")
    @patch("PIL.Image.open")
    @patch("os.path.exists")
    @patch("os.remove")
    def test_generate_video_ai_embeddings(self, mock_remove, mock_exists, mock_image_open, mock_extract):
        mock_extract.return_value = ["frame0.jpg", "frame1.jpg"]
        mock_exists.return_value = True
        
        # Mock Image.open context manager
        mock_img = MagicMock()
        mock_image_open.return_value.__enter__.return_value = mock_img
        
        # Mock preprocess output (tensors)
        mock_tensor = torch.zeros((3, 224, 224))
        mock_preprocess.return_value = mock_tensor
        
        # Mock model.encode_image to return dummy feature vector
        # ViT-B-32 outputs 512-dim features
        dummy_features = torch.ones((2, 512))
        mock_model.encode_image.return_value = dummy_features

        res = generate_video_ai_embeddings("dummy.mp4")
        
        self.assertEqual(res["video_path"], "dummy.mp4")
        self.assertEqual(len(res["ai_embeddings"]), 2)
        # Verify model.encode_image was called
        mock_model.encode_image.assert_called_once()
        # Verify cleanup
        mock_remove.assert_has_calls([call("frame0.jpg"), call("frame1.jpg")], any_order=True)

    def test_compare_ai_fingerprints_match(self):
        # We test Cosine Similarity using real PyTorch (runs on CPU/device)
        # Let's create two fingerprints that are identical: 2 frames of 512 dimensions
        # Embedding vectors are normalized to unit length in generate_video_ai_embeddings
        # So we create normalized vectors: e.g. [1.0, 0.0, ...]
        v1 = [0.0] * 512
        v1[0] = 1.0
        
        fp1 = [v1, v1]
        fp2 = [v1, v1]
        
        res = compare_ai_fingerprints(fp1, fp2, threshold=0.9, min_match_rate=0.9)
        self.assertTrue(res["is_ai_duplicate"])
        self.assertEqual(res["match_percentage"], 100.0)
        self.assertEqual(res["total_matched_frames"], 2)
        self.assertEqual(len(res["matched_segments"]), 2)
        self.assertAlmostEqual(res["matched_segments"][0]["cosine_similarity"], 1.0, places=4)

    def test_compare_ai_fingerprints_no_match(self):
        # Orthogonal vectors: cosine similarity = 0.0
        v1 = [0.0] * 512
        v1[0] = 1.0
        
        v2 = [0.0] * 512
        v2[1] = 1.0
        
        fp1 = [v1]
        fp2 = [v2]
        
        res = compare_ai_fingerprints(fp1, fp2, threshold=0.85, min_match_rate=0.9)
        self.assertFalse(res["is_ai_duplicate"])
        self.assertEqual(res["match_percentage"], 0.0)
        self.assertEqual(res["total_matched_frames"], 0)

    def test_compare_ai_fingerprints_empty(self):
        res = compare_ai_fingerprints([], [])
        self.assertFalse(res["is_ai_duplicate"])
        self.assertEqual(res["match_percentage"], 0.0)

if __name__ == "__main__":
    unittest.main()
