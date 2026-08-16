# Security & Privacy

## Data handling

- Uploaded PDFs are processed by the application for extraction and retrieval.
- The project does not require a paid cloud AI API or API key.
- Optional Ollama inference is local to the machine running Ollama.
- Do not upload confidential documents to a public deployment unless you control its storage and access configuration.

## Upload protections

- Only PDF uploads are accepted.
- Upload size is bounded by the application limit.
- PDF content is fingerprinted to avoid stale document indexing.

## Secrets

- Never commit `.env` files, API keys, access tokens, or credentials.
- Use `.env.example` as the configuration reference.

## Reporting a vulnerability

Please report security issues privately through GitHub rather than publishing sensitive exploit details in an issue.
