# Security Policy

## Supported Versions

We maintain and support the latest tagged release of **Lit2Vec** (e.g., `v1.0.0`) and its corresponding codebase.

| Version    | Supported          |
|------------|--------------------|
| `main`     | ✅ Always supported |
| `v1.0.x`   | ✅ Supported         |
| `< v1.0.0` | ❌ Not supported    |

---

## Reporting a Vulnerability

If you believe you’ve found a **security vulnerability**, **data leak**, or **license compliance issue** in the Lit2Vec dataset or codebase:

- Please **do not** open a public GitHub issue.
- Instead, email the maintainers privately at:

📫 **[security@yourdomain.org](mailto:security@yourdomain.org)**

We will acknowledge receipt within **72 hours** and aim to provide a full response within **7 days**.

---

## Scope

This project has the following security surface:

### ✅ In Scope
- Inadvertent inclusion of non-CC-licensed or NoDerivatives (ND) content
- Vulnerabilities in chunking/embedding pipelines that could expose private or unpublished data
- Malicious code injection via dependencies (e.g., eval in markdown)
- Disclosure of sensitive information in record metadata (e.g., emails, ORCIDs, author notes)
- Insecure default configs (e.g., unrestricted ports or open APIs)

### ❌ Not in Scope
- Vulnerabilities in upstream models (e.g., E5, DistilBART)
- Biases or hallucinations from pseudo-labels or GPT summaries
- Misuse of embeddings or dataset by downstream users

---

## License and Compliance

All included data has been:
- Collected only from sources with explicit **CC licenses that allow derivatives**
- Processed to remove any **CC-BY-ND** or **all rights reserved** content
- Annotated with `license_final` and `license_allows_derivatives` fields per record

We encourage users to rerun `scripts/license_audit.py` for downstream due diligence.

---

## Responsible Disclosure Timeline

1. Submit report via [security@yourdomain.org](mailto:security@yourdomain.org)
2. We validate, patch, and regenerate any affected release(s)
3. If the issue affects previously published versions, we notify:
   - Zenodo (to deprecate the version)
   - Hugging Face (if mirrored)
   - Journal publisher (if applicable)
4. Optional public disclosure with CVE if warranted

---

Thank you for helping us keep Lit2Vec and the open science community secure and legally sound.
