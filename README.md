# AI197J

A Streamlit application for document intelligence and AI-assisted analysis. Upload documents, build searchable context, chat with an LLM grounded in that context, and generate audio briefings — all behind user authentication with a Snowflake backend.

## Features

- **💬 Chatbot** — Conversational interface backed by multiple LLM providers (Anthropic Claude, OpenAI), with chat history persisted in Snowflake.
- **📁 Context Files** — Upload and parse documents (via LlamaParse), chunk and embed their contents, run Snowflake Cortex search, and build/inspect a knowledge graph of the extracted entities.
- **🎵 Audio Creator** — Generate a narration script from your content and synthesize speech with ElevenLabs.
- **🔐 Authentication** — Login/registration via `streamlit-authenticator`; per-user data and Snowflake session scoping.

## Tech stack

- **UI:** Streamlit (multipage navigation, custom theme)
- **LLMs / NLP:** Anthropic, OpenAI, Cohere, LangChain, LlamaIndex / LlamaParse, tiktoken
- **Data / search:** Snowflake (incl. Cortex search), Pinecone
- **Audio:** ElevenLabs
- **Graph:** streamlit-agraph
- **Other integrations:** AWS (Secrets Manager via boto3), FactSet

## Project structure

```
app.py                  # Entry point: auth, navigation, page wiring
auth/                   # Authentication manager (streamlit-authenticator)
pages/                  # Chatbot, Context Files, Audio Creator UI
chatbot/                # Chat manager + Snowflake-backed chat repository
context/                # Parsing, chunking, summary, Cortex search, knowledge graph
connectors/             # Snowflake, AWS, Cohere, Pinecone, FactSet clients
llm/                    # LLM factory / manager / repository abstractions
utils/                  # Chunkers, query handling, secrets retrieval
interfaces/             # Analysis & knowledge-graph interfaces
setup_tables.sql        # Snowflake schema setup
requirements.txt        # Python dependencies
```

## Local setup

Requires Python 3.11+.

```bash
git clone https://github.com/jjgodbout/AI197J.git
cd AI197J

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env             # then fill in your keys
streamlit run app.py
```

AWS credentials are resolved through the boto3 default chain (`~/.aws/credentials`,
environment variables, or an IAM role). The app reads application secrets from AWS
Secrets Manager (`document_pipeline`, region `us-east-1`).

## Configuration & secrets

All credentials are supplied at runtime — nothing is committed.

- **Local:** copy `.env.example` to `.env` and fill in values. `.env` is gitignored.
- **Streamlit Cloud:** set the same keys in the app's **Settings → Secrets**, using
  `.streamlit/secrets.toml.example` as the template. Streamlit exposes these as both
  `st.secrets` and environment variables, so the app's `os.getenv(...)` calls resolve
  them automatically. Be sure to include `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
  and `AWS_DEFAULT_REGION` so Secrets Manager is reachable.

## Deploying to Streamlit Cloud

1. On [share.streamlit.io](https://share.streamlit.io), create a new app from this repo.
2. **Repository:** `jjgodbout/AI197J` · **Branch:** `master` · **Main file:** `app.py`
3. Add your secrets under **Advanced settings → Secrets** (see above).
4. Deploy.

## License

Proprietary — all rights reserved.
