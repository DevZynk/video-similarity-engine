import unittest
from unittest.mock import patch, MagicMock
import os
import sys
import tempfile
from io import BytesIO

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lib.temp_file import temp_file, get_temp_path, delete_temp_file

class TestTempFile(unittest.TestCase):

    @patch("urllib.request.urlopen")
    def test_temp_file_url(self, mock_urlopen):
        # Mock URL response
        mock_response = MagicMock()
        mock_response.read.side_effect = [b"mock url content", b""]
        # In python, shutil.copyfileobj calls read(length)
        # So we mock it by returning the bytes and then empty
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        # Call temp_file
        path = temp_file("url", "http://example.com/video.mp4")
        
        self.assertTrue(os.path.exists(path))
        with open(path, "rb") as f:
            content = f.read()
        self.assertEqual(content, b"mock url content")
        
        # Cleanup
        delete_temp_file(path)

    def test_temp_file_blob_bytes(self):
        data = b"blob binary data"
        path = temp_file("blob", data)
        self.assertTrue(os.path.exists(path))
        with open(path, "rb") as f:
            content = f.read()
        self.assertEqual(content, data)
        delete_temp_file(path)

    def test_temp_file_blob_file_like(self):
        data = BytesIO(b"file like data")
        path = temp_file("blob", data)
        self.assertTrue(os.path.exists(path))
        with open(path, "rb") as f:
            content = f.read()
        self.assertEqual(content, b"file like data")
        delete_temp_file(path)

    def test_temp_file_blob_invalid(self):
        with self.assertRaises(ValueError):
            temp_file("blob", 12345) # invalid type

    def test_temp_file_file_exists(self):
        # Create a real source file
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"source file data")
            src_path = f.name
            
        try:
            path = temp_file("file", src_path)
            self.assertTrue(os.path.exists(path))
            with open(path, "rb") as f:
                content = f.read()
            self.assertEqual(content, b"source file data")
            delete_temp_file(path)
        finally:
            if os.path.exists(src_path):
                os.remove(src_path)

    def test_temp_file_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            temp_file("file", "non_existent_file_xyz.mp4")

    def test_temp_file_invalid_type(self):
        with self.assertRaises(ValueError):
            temp_file("invalid_type", None)

    def test_get_temp_path(self):
        mock_file = MagicMock()
        mock_file.name = "/path/to/mock/file"
        path = get_temp_path(mock_file)
        self.assertEqual(path, "/path/to/mock/file")
        mock_file.close.assert_called_once()

    def test_delete_temp_file(self):
        # Test deleting existing file
        with tempfile.NamedTemporaryFile(delete=False) as f:
            path = f.name
        self.assertTrue(os.path.exists(path))
        res = delete_temp_file(path)
        self.assertTrue(res)
        self.assertFalse(os.path.exists(path))

        # Test deleting non-existent file
        res = delete_temp_file("non_existent_file_xyz.mp4")
        self.assertFalse(res)

if __name__ == "__main__":
    unittest.main()
