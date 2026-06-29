import unittest
from unittest.mock import patch, mock_open
import os
import sys

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lib.config import merge_configs, load_config, DEFAULT_CONFIG

class TestConfig(unittest.TestCase):

    def test_merge_configs_basic(self):
        default = {"a": 1, "b": {"c": 2, "d": 3}}
        user = {"b": {"d": 4}, "e": 5}
        expected = {"a": 1, "b": {"c": 2, "d": 4}, "e": 5}
        result = merge_configs(default, user)
        self.assertEqual(result, expected)

    def test_merge_configs_non_dict_user(self):
        default = {"a": 1}
        user = None
        result = merge_configs(default, user)
        self.assertEqual(result, default)

    @patch("os.path.exists")
    def test_load_config_not_exist(self, mock_exists):
        mock_exists.return_value = False
        result = load_config()
        self.assertEqual(result, DEFAULT_CONFIG)

    @patch("os.path.exists")
    @patch("builtins.open", new_callable=mock_open, read_data="server:\n  port: 9000\n")
    def test_load_config_valid_yaml(self, mock_file, mock_exists):
        mock_exists.return_value = True
        result = load_config()
        self.assertEqual(result["server"]["port"], 9000)
        self.assertEqual(result["server"]["host"], DEFAULT_CONFIG["server"]["host"])

    @patch("os.path.exists")
    @patch("builtins.open", new_callable=mock_open, read_data="{invalid: yaml")
    def test_load_config_invalid_yaml(self, mock_file, mock_exists):
        mock_exists.return_value = True
        result = load_config()
        self.assertEqual(result, DEFAULT_CONFIG)

if __name__ == "__main__":
    unittest.main()
