# Deployment

## Interactive application

The application is a Python/Streamlit service. GitHub Pages cannot execute the Streamlit runtime; it is used only for the static project showcase.

### Free-first options

- **Recommended for a public demo:** deploy the repository from GitHub using Streamlit Community Cloud when an eligible free deployment is available.
- Run locally with `streamlit run app.py`.
- Run Ollama locally for optional LLM generation.

The codebase does not require a paid inference provider.

### Streamlit Community Cloud checklist

1. Push the repository to GitHub.
2. Open Streamlit Community Cloud and sign in with GitHub.
3. Create a new app and select `shivamsinghrajputtt/AI-PDF-Chat-Assistant`.
4. Set the main file to `app.py`.
5. Deploy.
6. Verify PDF upload, multi-document retrieval, grounded answers, source pages, reset chat, and the zero-cost fallback.

The application should work without secrets in its default deterministic mode. Ollama is intentionally local/optional and is not required for the public demo.

## GitHub Pages showcase

The repository's static showcase is for recruiter/portfolio viewing. It is already deployed through GitHub Actions.

Expected URL:

`https://shivamsinghrajputtt.github.io/AI-PDF-Chat-Assistant/`

## Production checklist

- [ ] Keep `.env` out of Git.
- [ ] Restrict upload size and validate PDF content.
- [ ] Use persistent storage only when required by the deployment host.
- [ ] Do not upload sensitive documents to a public demo.
- [ ] Run `pytest -q` before deployment.
- [ ] Verify the public demo does not expose uploaded document contents between users.
- [ ] Monitor memory/disk usage on free hosting tiers.
