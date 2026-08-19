# Security Policy

We take the security of `AMDLongContextServing` seriously and appreciate the
efforts of security researchers and users who responsibly disclose
vulnerabilities.

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues,
pull requests, or discussions.** Public disclosure of an unpatched
vulnerability puts all users at risk.

Instead, report it privately through one of the following channels:

1. **GitHub Private Vulnerability Reporting (preferred).** Open the repository's
   [**Security** tab](https://github.com/AMD-AGI/AMDLongContextServing/security)
   and click **Report a vulnerability**. This opens a private advisory visible
   only to the maintainers.
2. **AMD PSIRT.** If you cannot use GitHub or your report concerns AMD products
   more broadly, contact the AMD Product Security Incident Response Team at
   [psirt@amd.com](mailto:psirt@amd.com). See
   [AMD Product Security](https://www.amd.com/en/resources/product-security.html)
   for details.

Please include as much of the following as you can:

- A description of the vulnerability and its potential impact.
- Steps to reproduce (proof-of-concept, affected commit/version, configuration).
- Any relevant logs, stack traces, or screenshots.
- Whether the issue is already known publicly.

## What to Expect

- **Acknowledgement** of your report, typically within 5 business days.
- An initial assessment and, where applicable, a coordinated remediation plan.
- Updates on remediation progress at reasonable intervals.
- Credit for your responsible disclosure once a fix is released, if you wish.

We ask that you give us a reasonable amount of time to investigate and address
the issue before any public disclosure, and that you make a good-faith effort to
avoid privacy violations, data destruction, or service interruption during your
research.

## Scope

This policy covers the source code, build scripts, and benchmark tooling in this
repository. It does **not** cover vulnerabilities in upstream third-party
dependencies (e.g. vLLM, PyTorch, ROCm, AITER) — please report those to their
respective projects — though we welcome a heads-up so we can update pins or
mitigations on our side.
