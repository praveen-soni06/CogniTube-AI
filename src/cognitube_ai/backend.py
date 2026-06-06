import os
from urllib.parse import parse_qs, urlparse

import streamlit as st
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from youtube_transcript_api.proxies import GenericProxyConfig
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from youtube_transcript_api._errors import IpBlocked

load_dotenv()

parser = StrOutputParser()


def get_huggingface_token():
    return os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")


def has_huggingface_token():
    return bool(get_huggingface_token())


def get_youtube_proxy_url():
    return os.getenv("YOUTUBE_PROXY_URL")


@st.cache_resource
def get_llm():
    """Create the Hugging Face text generation model once per Streamlit session."""
    return HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-72B-Instruct",
        huggingfacehub_api_token=get_huggingface_token(),
        task="text-generation",
        max_new_tokens=512,
        temperature=0.3,
    )


def extract_video_id(url_or_id):
    """Extract a YouTube video ID from a URL, or return the input ID."""
    value = url_or_id.strip()

    if "youtube.com" not in value and "youtu.be" not in value:
        return value

    parsed = urlparse(value)

    if "youtu.be" in parsed.netloc:
        return parsed.path.strip("/").split("/")[0]

    query_video_id = parse_qs(parsed.query).get("v")
    if query_video_id:
        return query_video_id[0]

    path_parts = [part for part in parsed.path.split("/") if part]
    if len(path_parts) >= 2 and path_parts[0] in {"embed", "shorts", "live"}:
        return path_parts[1]

    return value


@st.cache_resource
def build_vector_store(transcript):
    """Create a cached FAISS vector store from a transcript."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])

    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=get_huggingface_token(),
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )

    return FAISS.from_documents(chunks, embeddings)


def get_transcript(video_id):
    """Fetch an English transcript for a YouTube video."""
    try:
        ytt_api = YouTubeTranscriptApi()
        transcript_list = ytt_api.fetch(video_id, languages=["en",'hi'])
        transcript = " ".join(chunk.text for chunk in transcript_list)

        if len(transcript) < 50:
            raise ValueError("Transcript is too short.")

        return transcript

    except IpBlocked:
        st.error(
            "YouTube blocked this IP/network. Try another network, or set "
            "YOUTUBE_PROXY_URL in your .env file."
        )
        return None

    except TranscriptsDisabled:
        st.error("No captions are available for this video.")
        return None

    except Exception as exc:
        st.error(f"Error: {exc}")
        return None
