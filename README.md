# CogniTube AI

[![Python](https://img.shields.io/badge/Python-3.9+-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-0.0+-0EA5E9?logo=chainlink&logoColor=white)](https://langchain.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Extract insights from YouTube videos instantly — ask questions, get answers from transcripts using AI-powered retrieval.**

## Overview

CogniTube AI is a Streamlit-based application that enables intelligent question-answering over YouTube video transcripts using a Retrieval-Augmented Generation (RAG) pipeline. Instead of manually scrolling through transcripts, users can simply enter a YouTube URL and ask natural language questions to get precise answers grounded in the video's actual content.

### Problem It Solves
Finding specific information in long YouTube videos is time-consuming. CogniTube AI automates this by:
- Fetching transcripts automatically from any YouTube video
- Indexing them for fast semantic search
- Answering user questions with context pulled directly from the transcript
- Requiring no GPU or local LLM setup — all inference runs on Hugging Face's cloud infrastructure

### Who It's For
- **Content Researchers** — quickly extract facts and quotes from educational videos
- **Students** — study videos more efficiently by asking specific questions
- **Analysts** — search multiple videos for comparable information
- **Developers** — a reference RAG implementation using LangChain, FAISS, and Hugging Face

## Key Features

- 🎬 **Easy URL Input** — paste a YouTube URL or video ID; automatic parsing handles multiple formats
- 📝 **Multi-language Transcripts** — fetches English or Hindi captions (automatic fallback)
- 🔒 **Proxy Support** — built-in proxy handling for networks where YouTube blocks transcript API access
- ⚡ **Fast Retrieval** — FAISS vector store with deterministic local embeddings (no external embedding API calls)
- 🤖 **AI-Powered Answers** — uses Hugging Face Inference API with Qwen2.5-72B (configurable)
- 💬 **Context-Aware** — retrieves top-6 most relevant transcript chunks and uses them to generate answers
- 🔐 **Simple Configuration** — environment variables only; no database setup required

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **Frontend** | Streamlit |
| **Backend** | Python |
| **Embeddings** | LocalHashEmbeddings (deterministic, local) |
| **Vector Store** | FAISS (CPU) |
| **LLM** | Hugging Face Inference API (Qwen2.5-72B-Instruct) |
| **Text Splitting** | LangChain RecursiveCharacterTextSplitter |
| **Transcript Source** | YouTube Transcript API |
| **Environment** | python-dotenv |

### Libraries
```
streamlit
python-dotenv
youtube-transcript-api
langchain
langchain-core
langchain-community
langchain-text-splitters
faiss-cpu
huggingface-hub
```

## Architecture & How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Interface                          │
│                    (Streamlit Web App)                          │
└────────────────┬────────────────────────────────────────────────┘
                 │
       ┌─────────▼─────────┐
       │  YouTube URL Input  │
       └─────────┬──────────┘
                 │
       ┌─────────▼──────────────────┐
       │  Extract Video ID           │
       │  (URL parser)              │
       └─────────┬──────────────────┘
                 │
       ┌─────────▼──────────────────┐
       │  YouTube Transcript API    │
       │  (Fetch en/hi captions)    │
       └─────────┬──────────────────┘
                 │
       ┌─────────▼──────────────────┐
       │  RecursiveCharacterSplitter│
       │  (chunk_size=1000,         │
       │   overlap=200)             │
       └─────────┬──────────────────┘
                 │
       ┌─────────▼──────────────────┐
       │  LocalHashEmbeddings       │
       │  (Deterministic, 384-dim)  │
       └─────────┬──────────────────┘
                 │
       ┌─────────▼──────────────────┐
       │  FAISS Vector Store        │
       │  (@st.cache_resource)      │
       └──────────┬─────────────────┘
                  │
        ┌─────────▼────────┐
        │  User Question   │
        └─────────┬────────┘
                  │
        ┌─────────▼──────────────────┐
        │  FAISS Retriever (k=6)     │
        │  Fetch top chunks          │
        └─────────┬──────────────────┘
                  │
        ┌─────────▼──────────────────┐
        │  Build Prompt              │
        │  (Context + Question)      │
        └─────────┬──────────────────┘
                  │
        ┌─────────▼──────────────────┐
        │  HF InferenceClient        │
        │  chat_completion           │
        │  (temp=0.3, max_tokens=512)│
        └─────────┬──────────────────┘
                  │
        ┌─────────▼────────────┐
        │  Answer to User      │
        └──────────────────────┘
```

## Project Structure

```
CogniTube-AI/
├── streamlit_app.py              # Entry point for Streamlit
├── src/
│   └── cognitube_ai/
│       ├── __init__.py           # Package init
│       ├── app.py                # UI logic and Streamlit components
│       └── backend.py            # Core functions: embeddings, vector store, LLM
├── requirements.txt              # Python dependencies
├── .env.example                  # Example environment variables
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

### File Descriptions

| File | Purpose |
|------|---------|
| `streamlit_app.py` | Adds `src/` to path and calls `cognitube_ai.app.main()` |
| `src/cognitube_ai/app.py` | Streamlit UI: input fields, buttons, state management |
| `src/cognitube_ai/backend.py` | Core logic: `LocalHashEmbeddings`, `build_vector_store()`, `get_transcript()`, `generate_answer()` |
| `.env.example` | Template for required environment variables |

## Installation

### Prerequisites
- **Python 3.9+**
- A Hugging Face account with API access (free tier available)

### Step-by-Step Setup

#### 1. Clone the Repository
```bash
git clone https://github.com/praveen-soni06/CogniTube-AI.git
cd CogniTube-AI
```

#### 2. Create and Activate Virtual Environment
**macOS/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 4. Configure Environment Variables
```bash
cp .env.example .env
```

Edit `.env` and add your Hugging Face token:
```env
HUGGINGFACEHUB_API_TOKEN=hf_your_actual_token_here
HUGGINGFACE_MODEL=Qwen/Qwen2.5-72B-Instruct
YOUTUBE_PROXY_URL=
```

You can get a free Hugging Face API token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

#### 5. Run the Application
```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`.

## Usage

### Basic Workflow

1. **Start the app:**
   ```bash
   streamlit run streamlit_app.py
   ```

2. **Enter a YouTube URL or Video ID:**
   - Full URL: `https://www.youtube.com/watch?v=dQw4w9WgXcQ`
   - Short URL: `https://youtu.be/dQw4w9WgXcQ`
   - Video ID: `dQw4w9WgXcQ`

3. **Click "Collect Transcript":**
   - The app fetches the transcript and builds a searchable vector store
   - Processing time depends on video length (typically 10-60 seconds)

4. **Ask a Question:**
   - Enter any question related to the video content
   - Click "Submit"
   - The app retrieves relevant sections and generates an answer

### Example Questions

For a tutorial video on Python:
- "What is list comprehension?"
- "How do I handle exceptions?"
- "What's the difference between == and is?"

## Configuration

All configuration is managed via `.env` file:

| Variable | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `HUGGINGFACEHUB_API_TOKEN` | string | ✅ Yes | — | Hugging Face API token for inference |
| `HF_TOKEN` | string | ❌ No | — | Alternative to `HUGGINGFACEHUB_API_TOKEN` |
| `HUGGINGFACE_MODEL` | string | ❌ No | `Qwen/Qwen2.5-72B-Instruct` | LLM model ID from Hugging Face |
| `YOUTUBE_PROXY_URL` | string | ❌ No | — | HTTP(S) proxy URL if YouTube blocks transcript fetching (e.g., `http://user:pass@proxy.com:8080`) |

### Obtaining a Hugging Face Token

1. Go to [huggingface.co](https://huggingface.co/)
2. Sign up or log in
3. Navigate to [Settings → API Tokens](https://huggingface.co/settings/tokens)
4. Click "New token" and create a token with "Read" access
5. Copy and paste into `.env`

## How the RAG Pipeline Works

### 1. Transcript Ingestion
- Fetches transcript via YouTube Transcript API
- Supports English (`en`) and Hindi (`hi`) with automatic fallback
- Validates minimum transcript length (50+ characters)

### 2. Text Chunking
- Uses `RecursiveCharacterTextSplitter` with:
  - Chunk size: 1000 characters
  - Overlap: 200 characters
- Preserves context across chunk boundaries

### 3. Embedding & Indexing
- **LocalHashEmbeddings**: Deterministic hash-based embeddings
  - Dimension: 384
  - Uses blake2b hashing on tokens
  - No external API calls required
  - Same text always produces the same embedding (reproducible)
- Indexed in FAISS for fast similarity search

### 4. Retrieval
- On question submit, retrieves top-6 most similar chunks
- Uses cosine similarity via FAISS

### 5. Generation
- Constructs a prompt combining:
  - System instruction (context-only reasoning)
  - Retrieved chunks as context
  - User question
- Sends to Hugging Face InferenceClient
- Parameters: `temperature=0.3`, `max_tokens=512`

## Limitations & Notes

- **Captions Required**: Only videos with captions (English or Hindi) are supported
- **Context Window**: Retrieves top-6 chunks (configurable in code)
- **Session-Based**: Vector stores are cached per Streamlit session; reload resets state
- **Rate Limits**: Depends on your Hugging Face API quota
- **Internet Required**: Requires connectivity to YouTube and Hugging Face APIs
- **No Persistent Storage**: Transcripts and vector stores are not saved between sessions

## Troubleshooting

### "YouTube blocked this IP/network"
**Solution**: Set `YOUTUBE_PROXY_URL` in `.env` to use a proxy:
```env
YOUTUBE_PROXY_URL=http://user:password@proxy.example.com:8080
```

### "No captions are available for this video"
**Solution**: Only videos with captions can be used. Check if the video has English or Hindi captions on YouTube.

### "Could not generate the answer"
**Solution**: 
- Verify your Hugging Face token is valid and has API access
- Check internet connectivity
- Ensure the model specified in `HUGGINGFACE_MODEL` exists and is accessible

### Empty or irrelevant answers
**Solution**: Try a more specific question or check that the video contains relevant content.

## Future Enhancements

Potential improvements (not currently implemented):

- [ ] Support for more languages (Spanish, French, German, etc.)
- [ ] Persistent transcript caching and session storage
- [ ] Support for local LLMs (Ollama, LM Studio)
- [ ] Custom embedding models (e.g., ONNX-based)
- [ ] Multi-video querying across playlists
- [ ] Docker containerization for easy deployment
- [ ] Streamlit Cloud deployment guide
- [ ] Unit tests and CI/CD pipeline
- [ ] Chat history and conversation context
- [ ] Adjustable retrieval parameters (k, temperature, chunk size) via UI

## Contributing

Contributions are welcome! Here's how to get started:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

### Development Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Create a feature branch
git checkout -b feature/my-feature

# Test your changes
streamlit run streamlit_app.py

# Submit a PR
```

## License

This project is not yet licensed. To use or contribute, please add a LICENSE file (e.g., MIT, Apache 2.0) to specify usage terms.

## Support

For issues, questions, or suggestions:
- Open an [issue on GitHub](https://github.com/praveen-soni06/CogniTube-AI/issues)
- Check existing issues first to avoid duplicates

## Acknowledgments

Built with:
- [Streamlit](https://streamlit.io/) — web app framework
- [LangChain](https://langchain.com/) — LLM orchestration
- [FAISS](https://github.com/facebookresearch/faiss) — vector search
- [Hugging Face](https://huggingface.co/) — hosted inference API
- [YouTube Transcript API](https://github.com/jderose9/youtube-transcript-api) — transcript access

---

**Made with ❤️ by [praveen-soni06](https://github.com/praveen-soni06)**
