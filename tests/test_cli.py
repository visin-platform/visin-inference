"""Exercise CLI paths without downloading or importing model packages."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from visin_infer import cli


class CliTest(unittest.TestCase):
    def test_wildcard_is_rejected_before_loading_models(self):
        with patch.object(cli, "resolve_weights") as resolve:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                cli.main(["serve", "--allow-origin", "*"])
        self.assertEqual(caught.exception.code, 2)
        resolve.assert_not_called()

    def test_fetch_with_disabled_models_does_not_download(self):
        with patch.object(cli, "resolve_weights") as resolve:
            self.assertEqual(cli.main(["fetch", "--detector", "none", "--segmenter", "none"]), 0)
        resolve.assert_not_called()

    def test_existing_weights_are_resolved_without_downloading(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "custom.pt"
            path.touch()
            self.assertEqual(cli.resolve_weights(str(path), Path(directory) / "unused"), str(path))
            self.assertEqual(cli.resolve_weights("custom.pt", Path(directory)), str(path))

    def test_model_free_server_passes_configuration_and_closes(self):
        server = Mock()
        server.serve_forever.side_effect = KeyboardInterrupt
        with (
            patch.object(cli, "create_server", return_value=server) as create,
            patch.object(cli, "Inference") as inference,
            patch.object(cli.secrets, "token_urlsafe", return_value="generated-token"),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            result = cli.main(
                [
                    "serve",
                    "--allow-origin",
                    "https://example.test",
                    "--detector",
                    "none",
                    "--segmenter",
                    "none",
                    "--port",
                    "8123",
                    "--token",
                ]
            )
        self.assertEqual(result, 0)
        inference.assert_called_once_with(None, None)
        create.assert_called_once_with(
            inference.return_value, ["https://example.test"], "127.0.0.1", 8123, "generated-token"
        )
        server.server_close.assert_called_once()
