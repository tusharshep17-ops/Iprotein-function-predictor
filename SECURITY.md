# Security policy

## Supported version

Security fixes are applied to the latest release on the `main` branch.

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting feature instead of opening a public
issue. Include reproduction steps, affected versions, and potential impact.

## Safe operation

- Never load an untrusted `.joblib`, `.pkl`, or pickle file. Deserialization may execute code.
- Keep downloaded sequences and research data private when required by their source policy.
- Do not put access tokens, credentials, or private API endpoints in the repository.
- The public UniProt downloader uses HTTPS, a request timeout, schema validation, sequence
  validation, and bounded per-page results.
- The Streamlit interface limits sequence length and batch size, but it is an educational
  local app and is not hardened as a multi-tenant production service.

