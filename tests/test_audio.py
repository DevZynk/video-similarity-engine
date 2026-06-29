import unittest
from unittest.mock import patch, MagicMock
import os
import sys
import acoustid

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lib.audio import generate_audio_fingerprint, compare_audio_fingerprints

class TestAudio(unittest.TestCase):

    @patch("acoustid.fingerprint_file")
    @patch("acoustid.chromaprint.decode_fingerprint")
    def test_generate_audio_fingerprint_success(self, mock_decode, mock_fingerprint):
        mock_fingerprint.return_value = (45.0, b"binary_fingerprint_data")
        mock_decode.return_value = ([12345, 67890], 1)

        res = generate_audio_fingerprint("dummy_audio.mp4")
        
        self.assertEqual(res["video_path"], "dummy_audio.mp4")
        self.assertEqual(res["duration"], 45.0)
        self.assertEqual(res["audio_subprints"], [12345, 67890])
        mock_fingerprint.assert_called_once_with("dummy_audio.mp4")
        mock_decode.assert_called_once_with(b"binary_fingerprint_data")

    @patch("acoustid.fingerprint_file")
    def test_generate_audio_fingerprint_error(self, mock_fingerprint):
        mock_fingerprint.side_effect = acoustid.AcoustidError("Library error")
        
        res = generate_audio_fingerprint("dummy_audio.mp4")
        self.assertEqual(res["video_path"], "dummy_audio.mp4")
        self.assertEqual(res["duration"], 0)
        self.assertEqual(res["audio_subprints"], [])

    def test_compare_audio_fingerprints_perfect_match(self):
        # We need subprints that are identical or very close.
        # Let's say fp1_list and fp2_list are identical.
        # Standard Chromaprint SUBPRINTS_PER_SECOND = 8.6
        # Let's create 10 subprints (about 1.16 seconds of audio)
        fp = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        
        # compare_audio_fingerprints(fp1, fp2, max_bit_diff)
        # Identical subprints should have 0 bit difference (XOR = 0, count('1') = 0)
        res = compare_audio_fingerprints(fp, fp, max_bit_diff=4)
        
        self.assertTrue(res["is_audio_duplicate"])
        self.assertEqual(res["match_percentage"], 100.0)
        self.assertEqual(res["total_matched_seconds"], 2) # 10 items / 8.6 = ~1.16, rounded/mapped:
        # sub_second for index 0 to 7: int(i / 8.6) + 1 = 1
        # sub_second for index 8 to 9: int(i / 8.6) + 1 = 2
        # So we have 2 distinct matched seconds (second 1 and second 2).

    def test_compare_audio_fingerprints_no_match(self):
        # Subprints with huge differences: val_sub ^ val_main has many 1 bits
        # e.g., 0 and 0xFFFFFFFF
        fp1 = [0] * 10
        fp2 = [0xFFFFFFFF] * 10
        
        res = compare_audio_fingerprints(fp1, fp2, max_bit_diff=4)
        self.assertFalse(res["is_audio_duplicate"])
        self.assertEqual(res["match_percentage"], 0.0)
        self.assertEqual(res["total_matched_seconds"], 0)
        self.assertEqual(res["matched_segments"], [])

    def test_compare_audio_fingerprints_empty(self):
        res = compare_audio_fingerprints([], [])
        self.assertFalse(res["is_audio_duplicate"])
        self.assertEqual(res["match_percentage"], 0.0)
        self.assertEqual(res["matched_segments"], [])

if __name__ == "__main__":
    unittest.main()
