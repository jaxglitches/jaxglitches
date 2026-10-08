# Development and releases

## Local checks

```sh
uv sync --locked --group docs --group examples --group release
uv run --no-sync pytest -q
uv run --no-sync python tools/build_example.py
uv run --no-sync mkdocs build --strict
uv build
uv run --no-sync twine check --strict dist/*
```

Preview the documentation with `uv run --no-sync mkdocs serve`.
CI runs the existing tests on Python 3.12 and 3.13, executes the example notebook,
and builds documentation strictly,
checks the wheel and source distribution metadata, and reruns tests against the
installed wheel. CUDA and scientific notebook campaigns are separate checks.

## GitHub Pages

The repository Pages source must be **GitHub Actions**. The `docs.yml` workflow
builds and deploys documentation when relevant files change on `main`, or when
manually dispatched. The site URL is
<https://jaxglitches.github.io/jaxglitches/> after a successful deployment.

## PyPI release

The trusted publisher is configured for:

- Project: `jaxglitches`
- Repository: `jaxglitches/jaxglitches`
- Workflow filename: `pypi.yml`
- Environment: unrestricted (the publish job does not specify an environment)

The workflow uses GitHub OIDC and needs no API token. It runs only when a GitHub
release is published, and waits for tests, documentation, and distribution checks.

1. Review the package metadata and MIT license (`LICENSE`), matching global-fit's
   license choice.
2. Set the version in `pyproject.toml`, update `uv.lock` with `uv lock`, and commit
   the changes and workflows to the repository.
3. Confirm CI passes for the release commit.
4. Publish a GitHub release with tag `v<version>` pointing to that commit.
   For version `0.1.0`, use `v0.1.0`.
5. Check the **Publish to PyPI** workflow and verify the uploaded project.

A mismatched tag fails before publishing. Each PyPI version can only be uploaded
once; prepare a new version for later releases. PyPI Trusted Publishing is
configured in the PyPI account, independently of GitHub Pages.

References: [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/using-a-publisher/)
and [GitHub Pages workflows](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).
