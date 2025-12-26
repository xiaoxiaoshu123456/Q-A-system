import tempfile
import unittest
from pathlib import Path
from upload_manifest import load_manifest, save_manifest, update_manifest_name, md5_hex, find_by_hash, manifest_path


class UploadManifestTests(unittest.TestCase):
    def test_save_and_load_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            items = [{"文件": "a.txt", "状态": "成功", "grandpa_id": "gid1", "时间": "now", "hash": "h1"}]
            save_manifest(root, items)
            self.assertTrue(manifest_path(root).exists())
            loaded = load_manifest(root)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0]["文件"], "a.txt")

    def test_update_manifest_name_multiple(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            gid = "gid_same"
            items = [
                {"文件": "(未知)", "状态": "已同步", "grandpa_id": gid, "时间": "t1"},
                {"文件": "(未知)", "状态": "已同步", "grandpa_id": gid, "时间": "t2"},
                {"文件": "b.txt", "状态": "成功", "grandpa_id": "gid_other", "时间": "t3"},
            ]
            save_manifest(root, items)
            update_manifest_name(root, gid, "auto_name")
            loaded = load_manifest(root)
            names = [i["文件"] for i in loaded if i.get("grandpa_id") == gid]
            self.assertTrue(all(n == "auto_name" for n in names))

    def test_md5_and_find_by_hash(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            h = md5_hex(b"abc")
            items = [{"文件": "a.txt", "状态": "成功", "grandpa_id": "gid1", "时间": "now", "hash": h}]
            save_manifest(root, items)
            loaded = load_manifest(root)
            found = find_by_hash(loaded, h)
            self.assertIsNotNone(found)
            self.assertEqual(found["文件"], "a.txt")


if __name__ == "__main__":
    unittest.main()

