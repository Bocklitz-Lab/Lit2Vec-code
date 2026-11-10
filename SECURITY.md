# Security Policy

## Supported versions

There is no tagged release yet. Security fixes are made on the `main` branch.

## Reporting a vulnerability

If you think you have found a security vulnerability, a data leak, or a license compliance problem in the code or in the released materials, please do not open a public GitHub issue. Write to the maintainers at **mahmoud.amiri@uni-jena.de** instead.

We will acknowledge your message within 72 hours and aim to send a full response within 7 days.

## Scope

In scope:

- A record in the released materials whose license was resolved wrongly, for example no-derivatives (ND) content that passed the screening.
- Code in the chunking, normalisation or embedding pipeline that could expose private or unpublished data, or execute content from a processed document.
- Dependencies with known vulnerabilities that this project pins.
- Personal information in the released materials, such as email addresses in record metadata.
- API keys or other credentials committed to the repository or to the Hugging Face repositories.

Out of scope:

- Vulnerabilities in upstream models such as e5-large-v2 or DistilBART.
- Biases or errors in GPT-4o-generated pseudo-labels and summaries.
- Misuse of the code, models or data by downstream users.

## Licenses and compliance

License screening in Lit2Vec is based on metadata from Unpaywall, OpenAlex and Crossref. It is an auditable filter, not legal clearance, and upstream records can be wrong or can change. The corpus itself is not redistributed. If you find a record that should not have passed the screening, or a released file that should not be public, please report it as described above. The [README](README.md) explains the screening rule and the licenses of the released materials.

## What happens after a report

1. We reproduce the problem and decide how serious it is.
2. We fix it and, where the released materials are affected, update or withdraw the affected files on GitHub and Hugging Face.
3. We credit you in the fix if you wish, and we publish a short note when the issue affected users.
