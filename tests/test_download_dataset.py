import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import httpx2
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, LocalTokenNotFoundError
from scripts.tools.download_dataset import main, DATASET_URL, REPO_ID


class DatasetDownloadTests(unittest.TestCase):
    def run_error(self, error, stage='auth_check'):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'data'
            target.mkdir()
            (target / 'existing.txt').write_text('keep')
            with patch('huggingface_hub.HfApi.auth_check') as auth, patch('huggingface_hub.snapshot_download') as download:
                (auth if stage == 'auth_check' else download).side_effect = error
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr):
                    code = main(['--output', str(target)])
                self.assertEqual((target / 'existing.txt').read_text(), 'keep')
                if stage == 'auth_check':
                    download.assert_not_called()
                return code, stderr.getvalue()

    def test_auth_errors_explain_request_approval_and_login(self):
        errors = [LocalTokenNotFoundError('Missing'),
                  GatedRepoError('Pending or rejected', response=httpx2.Response(403, request=httpx2.Request('GET', DATASET_URL))),
                  HfHubHTTPError('Unauthorized', response=httpx2.Response(401, request=httpx2.Request('GET', DATASET_URL))),
                  HfHubHTTPError('Forbidden', response=httpx2.Response(403, request=httpx2.Request('GET', DATASET_URL)))]
        for stage in ['auth_check', 'download']:
            for error in errors:
                with self.subTest(stage=stage, error=type(error).__name__):
                    code, message = self.run_error(error, stage)
                    self.assertEqual(code, 2)
                    for text in [DATASET_URL, 'aprobación manual', 'hf auth login', 'misma cuenta', 'no comercial', 'no identificar']:
                        self.assertIn(text, message)

    def test_unrelated_http_failure_is_not_called_access_denial(self):
        code, message = self.run_error(HfHubHTTPError('Internal', response=httpx2.Response(500, request=httpx2.Request('GET', DATASET_URL))))
        self.assertEqual(code, 1)
        self.assertIn('HTTP 500', message)
        self.assertNotIn('solicitud', message)

    def test_approved_download_requires_token_and_preserves_revision(self):
        with patch('huggingface_hub.HfApi.auth_check') as auth, patch('huggingface_hub.snapshot_download', return_value='data-output') as download:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(['--output', 'data-output', '--revision', 'specific-commit']), 0)
            auth.assert_called_once_with(REPO_ID, repo_type='dataset', token=True)
            download.assert_called_once_with(REPO_ID, repo_type='dataset', token=True,
                                             local_dir=Path('data-output'), revision='specific-commit')
