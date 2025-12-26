import json
import tempfile
import unittest
from pathlib import Path
from conversation_log import append, log_path


class ConversationLogTests(unittest.TestCase):
    def test_append_creates_file_and_writes_jsonl(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            entry = {"trace_id": "t1", "question": "q", "answer": "a"}
            append(root, entry)
            p = log_path(root)
            self.assertTrue(p.exists())
            lines = p.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            obj = json.loads(lines[0])
            self.assertEqual(obj["trace_id"], "t1")
            self.assertEqual(obj["question"], "q")
            self.assertEqual(obj["answer"], "a")
            self.assertTrue("ts" in obj)


if __name__ == "__main__":
    unittest.main()

