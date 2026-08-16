# Deployment

## Interactive application

The application is a Python/Streamlit service. GitHub Pages cannot execute the Streamlit runtime; it is used only for the static project showcase.

### Free-first options

- Run locally with `streamlit run app.py`.
- Use a free Streamlit-compatible hosting tier when available.
- Run Ollama locally for optional LLM generation.

No paid inference provider is required by the codebase.

## GitHub Pages showcase

The repository's static showcase is intended for recruiter/portfolio viewing. Enable GitHub Pages for the repository's Pages source/workflow to publish the static site.

Expected URL:

`https://shivamsinghrajputtt.github.io/AI-PDF-Chat-Assistant/`

## Production checklist

- [ ] Set repository secrets only when an external deployment provider requires them.
- [ ] Keep `.env` out of Git.
- [ ] Restrict upload size and validate PDF content.
- [ ] Use persistent storage only when required by the deployment host.
- [ ] Do not upload sensitive documents to a public demo.
- [ ] Run `pytest -q` before deployment.
