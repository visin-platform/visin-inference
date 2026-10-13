# Contributing

Use Python 3.10 or newer. Clone the repo and run `make install` to create a
virtual environment, install development dependencies and register pre-commit
hooks. The model stack is optional: `python -m pip install -e '.[models]'`.

Run `make check` before opening a pull request:

| Target | Check |
| --- | --- |
| `make lint` | Ruff lint and formatting; `make format` applies fixes. |
| `make typecheck` | Mypy, including bodies of untyped functions. |
| `make coverage` | Offline unit/HTTP tests, branch coverage report and 80% coverage floor. |
| `make build` | Wheel built from sdist and strict Twine metadata validation. |
| `make docs` | Strict MkDocs build. |

On Windows, run the commands from the Makefile using the activated environment,
or use a shell with Make. CI tests Linux Python 3.10–3.13, plus macOS and Windows
on 3.13. The protocol tests require local loopback sockets, NumPy and Pillow,
but no model downloads, GPU, Visin server, or sibling checkout.

Real-model tests are opt-in: install `[models]`, run `visin-infer fetch`, then
`make test-models`. They use Ultralytics' bundled example images and cached
weights. Explicit opt-in fails if prerequisites are missing, so missing models
cannot silently turn a requested integration run into a success.

Add tests for changed observable behavior. Keep model imports lazy and the
loopback default. Update the appropriate documentation and `CHANGELOG.md` for
user-facing changes. Conventional commit messages (`feat:`, `fix:`, `docs:`,
`test:`, `ci:`) follow the other Visin repositories' style.

See [release setup](https://github.com/visin-platform/visin-inference/blob/main/docs/releases.md) for maintainer publishing instructions and
[SECURITY.md](https://github.com/visin-platform/visin-inference/blob/main/SECURITY.md) for private vulnerability reports.
