import os
import base64
import json
import textwrap
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import main

ROOT = Path(__file__).resolve().parents[1]
CONTEXT_FILES = {
    'Dockerfile', 'Dockerfile.browser', 'requirements.txt', 'main.py',
    'agent_protocol.py', 'scripts/browser-entrypoint.sh',
    'scripts/browser_runtime.py', 'scripts/browser_probe.py',
}


class PackagingTests(unittest.TestCase):
    def test_context_is_default_deny_with_only_explicit_runtime_exceptions(self):
        rules = [line.strip() for line in (ROOT / '.dockerignore').read_text().splitlines()
                 if line.strip() and not line.startswith('#')]
        expected = ['**', '!Dockerfile', '!Dockerfile.browser', '!requirements.txt',
                    '!main.py', '!agent_protocol.py', '!scripts/', 'scripts/**',
                    '!scripts/browser-entrypoint.sh', '!scripts/browser_runtime.py',
                    '!scripts/browser_probe.py']
        self.assertEqual(rules, expected)

    def test_both_docker_recipes_only_copy_allowed_existing_files(self):
        for recipe in ['Dockerfile', 'Dockerfile.browser']:
            for line in (ROOT / recipe).read_text().splitlines():
                if line.startswith('COPY '):
                    for source in shlex.split(line)[1:-1]:
                        with self.subTest(recipe=recipe, source=source):
                            self.assertIn(source, CONTEXT_FILES)
                            self.assertTrue((ROOT / source).is_file())
                            self.assertFalse((ROOT / source).is_symlink())

    def test_cookie_import_is_not_an_implicit_image_dependency(self):
        self.assertNotIn('COOKIES_FILE', (ROOT / 'Dockerfile').read_text())

    def test_ci_checks_context_before_publishing_and_imports_before_deploy(self):
        workflow = (ROOT / '.github/workflows/deploy.yaml').read_text()
        self.assertLess(workflow.index('run: bash scripts/check-build-context.sh'),
                        workflow.index('uses: docker/build-push-action'))
        self.assertIn('load: true', workflow)
        self.assertIn('docker run --rm --pull=never', workflow)
        self.assertIn('image_version: ${{ steps.image.outputs.digest }}', workflow)
        self.assertLess(workflow.index('Bridge image import smoke test'),
                        workflow.index('name: Deploy to Socrat'))

    def test_ci_registry_credentials_are_private_ephemeral_and_never_printed(self):
        workflow = (ROOT / '.github/workflows/deploy.yaml').read_text()
        self.assertNotIn('docker/login-action', workflow)
        self.assertNotIn('${{ runner.temp }}', workflow)
        self.assertLess(workflow.index('Select ephemeral Docker config'), workflow.index('Configure Docker Buildx'))
        self.assertIn('DOCKER_CONFIG=$RUNNER_TEMP/venice-registry-auth', workflow)
        self.assertIn('if: always()\n        run: rm -f "$RUNNER_TEMP/venice-registry-auth/config.json"', workflow)
        script = textwrap.dedent(workflow.split("python - <<'PYCODE'\n", 1)[1].split('          PYCODE', 1)[0])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'private-auth'
            env = dict(os.environ, DOCKER_CONFIG=str(path), REGISTRY_USERNAME='synthetic-user', REGISTRY_PASSWORD='synthetic-private-canary')
            result = subprocess.run([sys.executable, '-c', script], env=env, capture_output=True, text=True, check=True)
            self.assertEqual(result.stdout + result.stderr, '')
            self.assertEqual(path.stat().st_mode & 0o777, 0o700)
            config = path / 'config.json'
            self.assertEqual(config.stat().st_mode & 0o777, 0o600)
            self.assertEqual(json.loads(config.read_text()), {'auths': {'harbor.socrat.alchemmist.xyz': {'auth': base64.b64encode(b'synthetic-user:synthetic-private-canary').decode()}}})
            again = subprocess.run([sys.executable, '-c', script], env=env, capture_output=True, text=True)
            self.assertNotEqual(again.returncode, 0)
            self.assertNotIn('synthetic-private-canary', again.stderr)
            self.assertNotIn('private-auth', CONTEXT_FILES)

    def test_readme_documents_mode_specific_environment(self):
        readme = (ROOT / 'README.md').read_text()
        for name in ['CDP_URL', 'PROFILE_DIR', 'PROXY_SERVER', 'HEADLESS', 'HOST',
                     'PORT', 'API_KEY', 'COOKIES_FILE', 'XDG_DATA_HOME', 'DEBUG_DIR',
                     'context_length_exceeded', 'history_validation']:
            with self.subTest(name=name):
                self.assertIn(name, readme)


class StandaloneSettingsTests(unittest.IsolatedAsyncioTestCase):
    async def test_no_cookie_import_without_explicit_opt_in(self):
        session = main.BrowserSession()
        session._ctx = MagicMock()
        session._ctx.add_cookies = AsyncMock()
        with patch.dict(os.environ, {'COOKIES_FILE': ''}), patch.object(Path, 'is_file') as exists:
            await session._apply_saved_cookies()
        exists.assert_not_called()
        session._ctx.add_cookies.assert_not_awaited()

    async def test_explicit_missing_cookie_file_fails_clearly(self):
        session = main.BrowserSession()
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {'COOKIES_FILE': str(Path(directory) / 'missing.json')}):
                with self.assertRaisesRegex(RuntimeError, 'Explicit COOKIES_FILE does not exist'):
                    await session._apply_saved_cookies()

    async def test_explicit_cookie_file_still_supported(self):
        session = main.BrowserSession()
        session._ctx = MagicMock()
        session._ctx.add_cookies = AsyncMock()
        with tempfile.TemporaryDirectory() as directory:
            cookie_file = Path(directory) / 'cookies.json'
            cookie_file.write_text('{"cookies":[{"name":"synthetic","value":"not-a-secret","domain":"venice.ai"}]}')
            with patch.dict(os.environ, {'COOKIES_FILE': str(cookie_file)}):
                await session._apply_saved_cookies()
        session._ctx.add_cookies.assert_awaited_once()

    async def test_new_dedicated_standalone_profile_is_private(self):
        session = main.BrowserSession()
        context = MagicMock()
        context.add_init_script = AsyncMock()
        pw = MagicMock()
        pw.chromium.launch_persistent_context = AsyncMock(return_value=context)
        manager = MagicMock()
        manager.start = AsyncMock(return_value=pw)
        session._apply_saved_cookies = AsyncMock()
        with tempfile.TemporaryDirectory() as directory:
            profile = Path(directory) / 'profile'
            profile.mkdir(mode=0o755)
            with patch.dict(os.environ, {'CDP_URL': ''}), \
                 patch.object(main, 'PROFILE_DIR', str(profile)), \
                 patch.object(main, 'CHROME_PROFILE', ''), \
                 patch.object(main, 'async_playwright', return_value=manager):
                await session._launch()
            self.assertEqual(profile.stat().st_mode & 0o777, 0o700)

    def test_default_profile_is_outside_checkout_and_respects_xdg(self):
        with tempfile.TemporaryDirectory() as directory:
            env = dict(os.environ, XDG_DATA_HOME=directory)
            env.pop('PROFILE_DIR', None)
            result = subprocess.run([sys.executable, '-c', 'import main; print(main.PROFILE_DIR)'],
                                    cwd=ROOT, env=env, text=True, capture_output=True, check=True)
            self.assertEqual(result.stdout.strip(), str((Path(directory) / 'venice-api/profile').resolve()))
            self.assertFalse((Path(directory) / 'venice-api/profile').exists())

    def test_empty_profile_setting_does_not_use_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            env = dict(os.environ, XDG_DATA_HOME=directory, PROFILE_DIR='')
            result = subprocess.run([sys.executable, '-c', 'import main; print(main.PROFILE_DIR)'],
                                    cwd=ROOT, env=env, text=True, capture_output=True, check=True)
            self.assertEqual(result.stdout.strip(), str((Path(directory) / 'venice-api/profile').resolve()))

    async def test_failed_cookie_initialization_cannot_leave_reusable_context(self):
        session = main.BrowserSession()
        context = MagicMock()
        context.add_init_script = AsyncMock()
        context.close = AsyncMock()
        pw = MagicMock()
        pw.chromium.launch_persistent_context = AsyncMock(return_value=context)
        pw.stop = AsyncMock()
        manager = MagicMock()
        manager.start = AsyncMock(return_value=pw)
        session._apply_saved_cookies = AsyncMock(side_effect=RuntimeError('Explicit cookie import failed'))
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {'CDP_URL': ''}), \
                 patch.object(main, 'PROFILE_DIR', str(Path(directory) / 'profile')), \
                 patch.object(main, 'CHROME_PROFILE', ''), \
                 patch.object(main, 'async_playwright', return_value=manager):
                with self.assertRaisesRegex(RuntimeError, 'cookie import failed'):
                    await session._launch()
        context.close.assert_awaited_once()
        pw.stop.assert_awaited_once()
        self.assertIsNone(session._ctx)
        self.assertIsNone(session._pw)
