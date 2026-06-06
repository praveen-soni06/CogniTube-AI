# CogniTube AI

CogniTube AI is a Streamlit app that answers questions from a YouTube video's transcript using a RAG pipeline.

## Features

- Accepts a YouTube URL or video ID
- Fetches the English transcript
- Splits transcript text into chunks
- Builds a FAISS vector store
- Answers questions using Hugging Face hosted inference

## Project Structure

```text
.
├── src/
│   └── cognitube_ai/
│       ├── __init__.py
│       ├── app.py
│       └── backend.py
├── streamlit_app.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Setup

Create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Create a `.env` file from `.env.example` and add your Hugging Face token:

```powershell
copy .env.example .env
```

Run the app:

```powershell
streamlit run streamlit_app.py
```

## Environment Variables

| Variable | Description |
| --- | --- |
| `HUGGINGFACEHUB_API_TOKEN` | Hugging Face access token used by LangChain's Hugging Face endpoint |

`HF_TOKEN` is also supported if you already use that variable name locally.
