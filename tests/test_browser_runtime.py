import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, MagicMock

spec = importlib.util.spec_from_file_location("runtime", Path(__file__).parents[1] / "scripts/browser_runtime.py")
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


class BrowserRuntimeTests(unittest.TestCase):
    def test_refuses_missing_proxy(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(SystemExit, "PROXY_SERVER"):
                runtime.main()

    def test_refuses_url_credentials(self):
        with patch.dict(os.environ, {"PROXY_SERVER": "http://user:secret@localhost:1080"}, clear=True):
            with self.assertRaisesRegex(SystemExit, "credentials"):
                runtime.main()

    def test_persistent_real_chrome_no_sandbox_bypass(self):
        with tempfile.TemporaryDirectory() as directory:
            profile = str((Path(directory) / "profile").resolve())
            process = MagicMock()
            process.wait.return_value = 0
            process.poll.return_value = 0
            with patch.dict(os.environ, {"PROXY_SERVER": "socks5://127.0.0.1:1080", "PROFILE_DIR": profile}, clear=True), patch.object(runtime.shutil, "which", return_value="/usr/bin/google-chrome-stable"), patch.object(runtime.subprocess, "Popen", return_value=process) as popen, patch.object(runtime.signal, "signal"):
                self.assertEqual(runtime.main(), 0)
            args = popen.call_args.args[0]
            self.assertIn("--user-data-dir=" + profile, args)
            self.assertIn("--remote-debugging-address=127.0.0.1", args)
            self.assertNotIn("--no-sandbox", args)
            self.assertFalse(any("headless" in arg or "AutomationControlled" in arg for arg in args))
            self.assertEqual(Path(profile).stat().st_mode & 0o777, 0o700)


if __name__ == "__main__":
    unittest.main()
