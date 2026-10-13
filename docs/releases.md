# Releases

## One-time repository setup

1. Enable Actions and create GitHub environments `pypi`, `testpypi` and
   `github-pages`. Restrict production publishers to release tags and use required
   reviewers where appropriate.
2. On PyPI and TestPyPI, configure trusted publishers for project `visin-infer`,
   owner `visin-platform`, repository `visin-inference`, workflow `release.yml`,
   using environments `pypi` and `testpypi` respectively. Verify the package name
   is available or that you control the existing project. No API token is needed.
3. Set GitHub Pages source to **GitHub Actions**. Enable private vulnerability
   reporting. Configure branch protection to require the CI jobs.
4. Confirm the existing AGPL-3.0-only license and rights to distribute all source
   contributions. Check dependency and model-weight license terms for your use.
   Model weights and local secrets are excluded from distributions.

Trusted publishing follows the [PyPI guide](https://docs.pypi.org/trusted-publishers/using-a-publisher/).
These account settings cannot be configured by checking in workflow files.

## Rehearse

Run **Release** manually on the intended branch or tag. It runs CI and publishes
its checked artifacts to TestPyPI only. Each TestPyPI upload needs a new version;
existing versions cannot be overwritten. No commit or tag is created.

## Publish

1. Update `visin_infer/__init__.py` (the sole version source) to `X.Y.Z`, move
   changelog entries into a release section, and run `make check`.
2. Merge the reviewed changes into main. Tag that commit `vX.Y.Z` and push the tag.
3. **Release** runs all CI checks on the tag, verifies the wheel version matches
   it, publishes the same wheel and sdist to PyPI, creates a GitHub release with
   artifacts and generated notes, and deploys the tagged documentation.

Tags must be `vMAJOR.MINOR.PATCH`; mismatches fail before any publishing. Avoid
moving a published tag. Releases use OIDC with job-scoped publishing permissions
and pinned third-party actions. Fork CI requires no secrets or write access.

**Docs** can also be dispatched manually for documentation fixes. It deploys the
selected ref. **Container build** builds and checks the Docker entry point for
relevant pull requests or manual runs; container registry publishing is not
configured. Build the image locally using the documented Docker commands.

If a publish succeeded but a later release/docs step failed, rerun only the
failed jobs; do not try to re-upload the already published PyPI version.
