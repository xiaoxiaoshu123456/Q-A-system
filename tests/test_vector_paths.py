import unittest
from pathlib import Path
from vector_store import VectorStorePaths


class VectorPathsTests(unittest.TestCase):
    def test_resolve_paths(self):
        paths = VectorStorePaths.resolve(config_path="config.ini", model_path=Path("model") / "bge-m3")
        self.assertTrue(paths.root.exists())
        self.assertEqual(paths.config_path.name, "config.ini")
        self.assertEqual(paths.model_path.name, "bge-m3")


if __name__ == "__main__":
    unittest.main()

