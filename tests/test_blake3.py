import unittest
import tempfile
import os
import sys

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lib.blake3 import blake3_file_fingerprint, blake3_compare_fingerprint

class TestBlake3(unittest.TestCase):

    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(delete=False)
        self.temp_file.write(b"hello world")
        self.temp_file.close()

    def tearDown(self):
        if os.path.exists(self.temp_file.name):
            os.remove(self.temp_file.name)

    def test_blake3_file_fingerprint(self):
        # The expected Blake3 hexdigest of "hello world"
        # Blake3 hash of "hello world" is: d74981efa70a0c20cffe40b6c68e1140696bd3a0c762589f50f576e273010b9a
        expected_hash = "d74981efa70a0c880b8d8c1985d075dbcbf679b99a5f9914e5aaf96b831a9e24"
        result_hash = blake3_file_fingerprint(self.temp_file.name)
        self.assertEqual(result_hash, expected_hash)

    def test_blake3_compare_fingerprint(self):
        h1 = "hash1"
        h2 = "hash1"
        h3 = "hash2"
        self.assertTrue(blake3_compare_fingerprint(h1, h2))
        self.assertFalse(blake3_compare_fingerprint(h1, h3))

if __name__ == "__main__":
    unittest.main()
