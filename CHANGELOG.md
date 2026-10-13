# Changelog

## [Unreleased]

- License repository source under MIT, matching the other Visin repositories; retain third-party dependency license notices.

- Add offline cross-platform CI, Ruff, Mypy, coverage reporting and pre-commit hooks.
- Add strict documentation builds and GitHub Pages deployment.
- Add tag-validated PyPI releases and manual TestPyPI rehearsals using trusted publishing.
- Keep the package version in `visin_infer.__version__` and include license metadata in distributions.
- Move Ultralytics to the `models` extra; install `visin-infer[models]` to run real inference.
- Make real-model tests explicitly opt-in and fail if an opted-in run lacks prerequisites.

## [0.1.0]

Initial local HTTP inference server with YOLO detection, SAM segmentation,
configurable origins, optional token authentication and VMSK mask responses.
