# CogniTube AI

CogniTube AI — A Streamlit YouTube RAG Chatbot

Short description: A Streamlit app that answers questions from a YouTube video's transcript using a Retrieval-Augmented Generation (RAG) pipeline.

Badges
- (No CI / license badges found in the repository. Add them if you enable CI or add a license.)

Overview
--------
What the project does
- Fetches a YouTube transcript (English or Hindi), splits it into chunks, builds a FAISS vector store with deterministic local embeddings, and answers user questions using a Hugging Face chat inference model.

Why it was built
- To provide a simple Streamlit-based interface that lets users ask questions about a YouTube video's transcript via a RAG-style approach without requiring remote embedding API calls.

Main problem it solves
- Quickly extract and query information from YouTube transcripts using vector search and an LLM for answer generation.

Key features
- Accepts a YouTube URL or video ID
- Fetches English or Hindi transcripts (youtube_transcript_api)
- Splits transcript text into chunks (RecursiveCharacterTextSplitter)
- Builds a FAISS vector store (langchain_community.vectorstores.FAISS)
- Uses deterministic local hash embeddings to avoid external embedding calls
- Answers questions using Hugging Face InferenceClient (chat completion)

Demo
----
- Live demo: (none provided in repository)
- Screenshots: (none present in repository)
- GIF: (none present in repository)
- Video demo: (none present in repository)

Placeholders — add your assets here:
- Live demo: https://your-live-demo.example
- Screenshots: docs/images/home.png
- GIF: docs/images/demo.gif
- Video: docs/video/demo.mp4

Features (detailed)
------------------
- Input: YouTube URL or plain video ID
- Transcript fetching: attempts English (`en`) and Hindi (`hi`) captions via youtube_transcript_api
- Proxy support for YouTube transcript API via YOUTUBE_PROXY_URL
- Text splitting: chunk_size=1000, chunk_overlap=200 (RecursiveCharacterTextSplitter)
- Local deterministic embeddings: LocalHashEmbeddings (class in src/cognitube_ai/backend.py)
  - Deterministic hashing using blake2b, normalized vector of fixed dimension (default 384)
- Vector store: FAISS (via langchain community vectorstores)
- Retrieval: vector_store.as_retriever(search_kwargs={"k": 6}) used in the Streamlit flow
- Answer generation: Hugging Face InferenceClient.chat_completion (max_tokens=512, temperature=0.3)
- Streamlit UI for collecting transcripts and submitting questions

Tech Stack
----------
- Frontend
  - Streamlit (single-app UI)
- Backend
  - Python code in src/cognitube_ai (app.py, backend.py)
- Machine Learning / AI
  - Hugging Face InferenceClient (remote chat completion)
  - LocalHashEmbeddings (custom deterministic embeddings implementation)
  - FAISS vector store (faiss-cpu)
  - LangChain components (text splitters, vector store integrations)
- Database
  - None (no persistent database; FAISS index built in-memory and cached per Streamlit session)
- APIs
  - YouTube transcript access via youtube_transcript_api (no public REST API endpoints in repo)
- Authentication
  - Uses a Hugging Face API token (HUGGINGFACEHUB_API_TOKEN or HF_TOKEN) for inference requests
- Deployment
  - Not specified in repository (no Dockerfile / no deployment scripts included)
- Programming Languages
  - Python (100% of repo)
- Libraries (from requirements.txt and code)
  - streamlit
  - python-dotenv
  - youtube-transcript-api
  - langchain
  - langchain-core
  - langchain-community
  - langchain-text-splitters
  - faiss-cpu
  - huggingface-hub
- Tools
  - dotenv for environment variable loading
  - huggingface_hub.InferenceClient

Project structure
-----------------
Repository top-level structure (actual)

```
.
|-- src/
|   `-- cognitube_ai/
|       |-- __init__.py
|       |-- app.py
|       `-- backend.py
|-- streamlit_app.py
|-- requirements.txt
|-- .env.example
|-- .gitignore
`-- README.md  (this file)
```

What each important folder/file contains
- src/cognitube_ai/app.py
  - Streamlit UI logic. Collects user input, triggers transcript collection, builds vector store, performs retrieval, builds prompt, and shows Hugging Face-generated answer.
- src/cognitube_ai/backend.py
  - Core helpers: LocalHashEmbeddings implementation, build_vector_store(), get_transcript(), extract_video_id(), get_huggingface_client(), generate_answer(), and proxy support for YouTube transcript fetching.
- streamlit_app.py
  - App entrypoint. Adjusts sys.path to include src and calls cognitube_ai.app.main().
- requirements.txt
  - Python package dependencies.
- .env.example
  - Example environment variables required to run the app.

Installation
------------
Follow these steps (commands assume a POSIX shell; Windows PowerShell examples are also included in the original README):

1. Clone repository
```bash
git clone https://github.com/praveen-soni06/CogniTube-AI.git
cd CogniTube-AI
```

2. Create a Python virtual environment and activate it
- macOS/Linux:
```bash
python -m venv venv
source venv/bin/activate
```
- Windows (PowerShell, taken from the repository README):
```powershell
python -m venv venv
.\\venv\\Scripts\\Activate.ps1
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Configure environment variables
- Copy .env.example -> .env and set your Hugging Face token:
```bash
cp .env.example .env
# then edit .env to set HUGGINGFACEHUB_API_TOKEN
```

5. Run the app
```bash
streamlit run streamlit_app.py
```

6. Verify installation
- In the Streamlit UI, enter a YouTube URL or video ID and click "Collect Transcript".
- After transcript processing, ask a question and click "Submit".

Environment Variables
---------------------
The repository contains `.env.example` with the following variables:

| Variable | Description | Required |
|----------|-------------|----------|
| HUGGINGFACEHUB_API_TOKEN | Hugging Face access token used for chat inference via huggingface_hub.InferenceClient | Yes |
| HUGGINGFACE_MODEL | Optional: Hugging Face model ID used by InferenceClient. Defaults to `Qwen/Qwen2.5-72B-Instruct` in code | No (optional) |
| YOUTUBE_PROXY_URL | Optional HTTP(S) proxy (e.g., `http://user:pass@host:port`) used when YouTube blocks your IP for transcript fetching | No (optional) |

Notes:
- The code also checks `HF_TOKEN` environment variable as an alternative to `HUGGINGFACEHUB_API_TOKEN`.
- The default model constant in code: `DEFAULT_LLM_MODEL = "Qwen/Qwen2.5-72B-Instruct"`

Requirements
------------
- Python version: not specified in the repository (no explicit Python version found).
- Node/npm: not used (no package.json).
- Database: none required.
- GPU / CUDA: not required by the repository (the code uses Hugging Face InferenceClient for remote inference and faiss-cpu). No local large-model inference is performed by the repository code.
- RAM / Disk: not specified.

Usage
-----
1. Start the Streamlit app:
```bash
streamlit run streamlit_app.py
```
2. In the UI:
  - Enter a YouTube URL or Video ID.
  - Click "Collect Transcript".
  - After the transcript is processed, enter a question in "Ask your question:" and click "Submit".
3. The app will retrieve the top-k documents from the vector store (k=6), build a prompt that contains only those docs, and call Hugging Face InferenceClient to generate an answer. If the context is insufficient, the assistant is instructed to say "I don't know."

API Documentation
-----------------
- No external REST API endpoints are present in this repository. The app is a Streamlit single-process UI application.
- Internal notable functions (defined in src/cognitube_ai/backend.py):
  - extract_video_id(url_or_id): parse YouTube URL or return given ID.
  - get_transcript(video_id): fetches transcript (languages en/hi) with proxy support.
  - build_vector_store(transcript): splits transcript and creates FAISS vector store with LocalHashEmbeddings.
  - generate_answer(prompt): calls Hugging Face InferenceClient.chat_completion to generate a response.

Database
--------
- No database is used. The vector store (FAISS) is built in memory and is cached per Streamlit session via @st.cache_resource.

Machine Learning / Model details
-------------------------------
- Embeddings: LocalHashEmbeddings (deterministic, local implementation in backend.py). It tokenizes text with a regex and maps tokens to a fixed-dimension vector (default 384) using blake2b hashing and normalization.
- Text splitting: RecursiveCharacterTextSplitter with chunk_size=1000 and chunk_overlap=200.
- Vector store: FAISS (faiss-cpu), built from the document chunks and LocalHashEmbeddings.
- LLM: Uses Hugging Face InferenceClient (Inference API) to perform chat completion. Default model set to Qwen/Qwen2.5-72B-Instruct in code; the model can be overridden via HUGGINGFACE_MODEL environment variable.
- Training: no training code included (the app performs retrieval and calls remote inference).
- Prediction flow:
  1. Build FAISS vector store from transcript chunks.
  2. On question submit: retrieve top-k docs (k=6).
  3. Format docs into prompt and call Hugging Face chat_completion.
  4. Display returned answer.
- Evaluation / accuracy: not provided in repository.

Configuration
-------------
Configurable settings are controlled by environment variables in `.env`:
- `HUGGINGFACEHUB_API_TOKEN` or `HF_TOKEN` — required to call the Hugging Face Inference API.
- `HUGGINGFACE_MODEL` — optional override of the LLM model.
- `YOUTUBE_PROXY_URL` — set when YouTube blocks transcript fetching from your IP; the code uses GenericProxyConfig for both http and https.

Screenshots
-----------
(Placeholders — no images in repo)

## Home Page

![Home](docs/images/home.png)

## Transcript collected

![Transcript](docs/images/transcript.png)

Workflow
--------
High-level flow from the code:

```mermaid
flowchart LR
  User --> StreamlitUI[Streamlit UI (streamlit_app.py / app.py)]
  StreamlitUI --> YouTubeTranscriptAPI[YouTube Transcript API (youtube_transcript_api)]
  YouTubeTranscriptAPI --> Transcript[Transcript text]
  Transcript --> TextSplitter[RecursiveCharacterTextSplitter]
  TextSplitter --> Chunks[Document chunks]
  Chunks --> LocalEmbeddings[LocalHashEmbeddings]
  LocalEmbeddings --> FAISS[FAISS vector store]
  User --> Query[Question]
  Query --> Retriever[FAISS retriever (k=6)]
  Retriever --> PromptBuilder[Build prompt with docs + instructions]
  PromptBuilder --> HFInference[Hugging Face InferenceClient (chat_completion)]
  HFInference --> Answer[Answer shown to User]
```

Notes & next steps
------------------
- The repository contains a working Streamlit app that uses a remote LLM (Hugging Face) and local deterministic embeddings with FAISS.
- Items you may want to add (not present currently):
  - LICENSE file (choose a license).
  - Explicit Python version (e.g., in README or runtime.txt).
  - Dockerfile for containerized deployment.
  - CI workflow (GitHub Actions) for linting/testing.
  - Tests for backend functions (extract_video_id, LocalHashEmbeddings, get_transcript).
  - Example screenshots or a demo link.

If you want, I can:
- Create or update README.md in the repository with the content above (I can commit it for you if you tell me the target repository and branch).
- Add a LICENSE file (if you tell me which license you want).
- Create a Dockerfile and a basic GitHub Actions workflow for running tests/lint (but I will need guidance about Python version, test framework, and desired deployment strategy).

---

Would you like me to commit this README.md to the repository (and if so, which branch should I use)?
