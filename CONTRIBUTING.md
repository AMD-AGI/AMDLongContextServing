# Contributing to AMDLongContextServing

Thanks for your interest in contributing! This repository reproduces the
Kimi-Linear long-context FP8 benchmark sweep on AMD Instinct™ MI355X hardware.
Contributions of all kinds are welcome — bug reports, reproduction results,
documentation fixes, and code improvements.

## Code of Conduct

Please be respectful and constructive in all interactions. By participating you
agree to uphold a harassment-free, professional environment for everyone.

## Ways to Contribute

- **Report a bug** or unexpected benchmark behavior by opening a
  [GitHub issue](https://github.com/AMD-AGI/AMDLongContextServing/issues).
- **Suggest an enhancement** (new sweep range, kernel option, report metric).
- **Submit a pull request** with a fix or improvement.

For anything security-related, do **not** open a public issue — follow
[`SECURITY.md`](SECURITY.md) instead.

## Reporting Issues

Before filing, please search existing issues to avoid duplicates. A good report
includes:

- What you ran (`make` target, `FROM`/`TO`/`REPEATS`, environment overrides).
- Hardware/software context (GPU, ROCm version, image tag, commit SHA).
- Expected vs. actual behavior, with relevant logs or the failing `report.md`.
- Minimal steps to reproduce.

## Development Setup

The canonical entry point is a single command from the repo root:

```bash
make run                              # full 1Ki..64Mi sweep
make run FROM=4Mi TO=8Mi REPEATS=3    # sub-range with repeats per point
```

See the [`README.md`](README.md) for prerequisites (Hugging Face token, Docker,
ROCm) and the available overrides. The benchmark runs inside a pinned Docker
image; please keep changes compatible with that image and the one-command flow.

## Pull Request Process

1. **Fork** the repository and create a topic branch from `main`
   (e.g. `fix/decode-report-rounding`).
2. **Make focused changes.** Keep PRs small and scoped to one logical change.
3. **Match the existing style.** Follow the conventions already present in the
   surrounding code; do not introduce unrelated reformatting.
4. **Test your change.** Where practical, validate against a short sweep range
   and include before/after evidence (numbers, `report.md`, or plots) in the PR
   description.
5. **Write a clear description.** Explain the motivation ("why") and summarize
   the change. Reference any related issues.
6. **Open the PR** against `main`. A maintainer / code owner review and passing
   checks are required before merge.

## Commit and Sign-off

- Use clear, imperative commit messages (e.g. "Fix 1Mi warm-up discard
  threshold").
- Optionally, sign off your commits with `git commit -s` to certify under the
  [Developer Certificate of Origin (DCO)](https://developercertificate.org/)
  that you have the right to submit the work under the project license.

## Licensing

This project is licensed under the **Apache License 2.0** (see
[`LICENSE`](LICENSE)). By contributing, you agree that your contributions will
be licensed under the same terms.

## External Contributors

This repository is part of the AMD-AGI GitHub organization. Non-AMD contributors
are welcome to open issues and submit pull requests from a fork. Adding an
external contributor as a repository collaborator requires admin approval and is
granted at the minimum necessary permission level.

## Questions

Open a [GitHub issue](https://github.com/AMD-AGI/AMDLongContextServing/issues)
or reach the maintainers via the owners listed in `CODEOWNERS`.
