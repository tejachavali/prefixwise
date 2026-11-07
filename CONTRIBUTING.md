# Contributing

Thanks for trying Prefixwise. Bug reports with a reproduction are the most useful contribution: a request body (with secrets removed), what Prefixwise said, and what you expected.

## Setup

```bash
git clone https://github.com/tejachavali/prefixwise && cd prefixwise
python -m pip install -e ".[dev]"
python -m pytest tests -q            # the whole suite runs in ~15 s, no network
python experiments/e1_open/e1_open.py --mock
```

The runtime uses the Python standard library only. Please keep it that way; test-only dependencies go in the `dev` extra.

## How changes are made

- **One PR per change**, with tests. CI runs the suite on Python 3.10 to 3.13, plus the E1 mock and a Docker build.
- **A classifier change needs a test with the exact request shapes that motivated it.** Shapes observed from real providers go in tests verbatim, with the date.
- **Experiments are evidence, not targets.** Never change code to make an E1 or E4 run pass. A failed run is a finding: record it under `experiments/<name>/runs/` and fix the cause. "Adjusted until green" proves nothing.
- **Read [`WORKLOG.md`](WORKLOG.md)** for what is in flight and why, and add an entry when you start something. `build.log` records outcomes.

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
4. `.github/workflows/release.yml` checks that the tag matches the version, runs the tests, builds, and publishes.

## Security

Please report vulnerabilities privately: see [SECURITY.md](SECURITY.md).
