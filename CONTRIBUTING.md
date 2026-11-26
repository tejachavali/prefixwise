# Contributing

Thanks for trying Prefixwise. Bug reports with a reproduction are the most useful contribution: a request body (with secrets removed), what Prefixwise said, and what you expected.

## Setup

```bash
git clone https://github.com/tejachavali/prefixwise && cd prefixwise
python -m pip install -e ".[dev]"
python -m pytest tests -q            # the whole suite: about 3 minutes, no network
python experiments/e1_open/e1_open.py --mock
```

The runtime uses the Python standard library only. Please keep it that way; test-only dependencies go in the `dev` extra.

## Working from a fork

You don't need write access: fork the repository, work on a branch in your fork, and open a pull request to
`main` here. CI runs on pull requests from forks too (a maintainer approves the first run).

```bash
git clone https://github.com/<you>/prefixwise && cd prefixwise
git remote add upstream https://github.com/tejachavali/prefixwise
python -m pip install -e ".[dev]" && python -m pytest tests -q

git fetch upstream && git checkout -b my-change upstream/main    # a fresh branch for each change
git push -u origin my-change                                      # then "Compare & pull request" on GitHub
git fetch upstream && git rebase upstream/main && git push --force-with-lease   # when main moves on
```

In the pull request, write `Closes #<issue>` and paste the test output. Leave "Allow edits by maintainers" on.

## How changes are made

- **One PR per change**, with tests. CI runs the suite on Python 3.10 to 3.13, plus the E1 mock and a Docker build.
- **A classifier change needs a test with the exact request shapes that motivated it.** Shapes observed from real providers go in tests verbatim, with the date.
- **Experiments are evidence, not targets.** Never change code to make an E1 or E4 run pass. A failed run is a finding: record it under `experiments/<name>/runs/` and fix the cause. "Adjusted until green" proves nothing.
- **Read [`docs/WORKLOG.md`](docs/WORKLOG.md)** for what is in flight and why, and add an entry when you start something. Run records under `experiments/*/runs/` and the CHANGELOG record outcomes.

## Releasing

One-time, on pypi.org: **Your projects → Publishing → Add a pending publisher**, with:
- project `prefixwise`
- owner `tejachavali`, repository `prefixwise`
- workflow `release.yml`
- environment `pypi`

Also create a GitHub environment named `pypi` (repo **Settings → Environments**). No API token is needed or stored.

Each release:
1. Set `version` in `pyproject.toml` and `__version__` in `src/prefixwise/__init__.py`.
2. Move the CHANGELOG's "unreleased" heading to the version and date, and merge to `main`.
3. Tag the merge: `git tag v0.3.0 && git push origin v0.3.0`.
4. `.github/workflows/release.yml` checks that the tag matches the version, runs the tests, builds, publishes to PyPI,
   then creates the GitHub Release from that version's CHANGELOG section with the same files (so the Releases page
   never lags PyPI). `tests/test_release.py` fails early if the CHANGELOG has no section for the current version.

## Security

Please report vulnerabilities privately: see [SECURITY.md](SECURITY.md).
