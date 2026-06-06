import os
import hashlib
import math
import re
from urllib.parse import parse_qs, urlparse

import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from huggingface_hub import InferenceClient
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from youtube_transcript_api import TranscriptsDisabled, YouTubeTranscriptApi
from youtube_transcript_api._errors import IpBlocked
from youtube_transcript_api.proxies import GenericProxyConfig

load_dotenv()

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+")
DEFAULT_LLM_MODEL = "Qwen/Qwen2.5-72B-Instruct"


class LocalHashEmbeddings(Embeddings):
    """Small deterministic local embeddings to avoid remote embedding API calls."""

    def __init__(self, dimensions=384):
        self.dimensions = dimensions

    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)

    def _embed(self, text):
        vector = [0.0] * self.dimensions
        tokens = TOKEN_PATTERN.findall(text.lower())

        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector

        return [value / norm for value in vector]


def get_huggingface_token():
    return os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")


def has_huggingface_token():
    return bool(get_huggingface_token())


def get_youtube_proxy_url():
    return os.getenv("YOUTUBE_PROXY_URL")


def get_youtube_client():
    proxy_url = get_youtube_proxy_url()

    if not proxy_url:
        return YouTubeTranscriptApi()

    return YouTubeTranscriptApi(
        proxy_config=GenericProxyConfig(
            http_url=proxy_url,
            https_url=proxy_url,
        )
    )


@st.cache_resource
def get_huggingface_client():
    """Create the Hugging Face inference client once per Streamlit session."""
    return InferenceClient(
        model=os.getenv("HUGGINGFACE_MODEL", DEFAULT_LLM_MODEL),
        token=get_huggingface_token(),
        timeout=120,
    )


def generate_answer(prompt):
    """Generate an answer with a Hugging Face chat/conversational model."""
    response = get_huggingface_client().chat_completion(
        messages=[
            {
                "role": "system",
                "content": (
                    "You answer questions using only the provided transcript context. "
                    "If the context is insufficient, say \"I don't know.\""
                ),
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=512,
        temperature=0.3,
    )

    return response.choices[0].message.content.strip()


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
    embeddings = LocalHashEmbeddings()

    return FAISS.from_documents(chunks, embeddings)


def get_transcript(video_id):
    """Fetch an English or Hindi transcript for a YouTube video."""
    try:
        ytt_api = get_youtube_client()
        transcript_list = ytt_api.fetch(video_id, languages=["en", "hi"])
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
