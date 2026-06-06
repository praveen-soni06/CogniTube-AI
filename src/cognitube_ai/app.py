import streamlit as st

from cognitube_ai.backend import (
    build_vector_store,
    extract_video_id,
    generate_answer,
    get_transcript,
    has_huggingface_token,
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_prompt(context, question):
    return f"""
You are a helpful assistant.
Answer ONLY from the provided transcript context.
If the context is insufficient, say "I don't know."

Context:
{context}

Question:
{question}

Answer:
""".strip()


def main():
    st.set_page_config(page_title="YouTube RAG Chatbot", layout="centered")
    st.title("YouTube RAG Chatbot")

    video_input = st.text_input("Enter YouTube URL or Video ID:")

    collect_clicked = st.button(
        "Collect Transcript",
        type="primary",
        use_container_width=True,
    )

    if collect_clicked:
        if not video_input:
            st.warning("Please enter a YouTube URL or video ID.")
            st.stop()

        video_id = extract_video_id(video_input)

        with st.spinner("Fetching transcript..."):
            transcript = get_transcript(video_id)

        if not transcript:
            st.stop()

        with st.spinner("Processing transcript..."):
            try:
                st.session_state.vector_store = build_vector_store(transcript)
                st.session_state.video_id = video_id
            except Exception as exc:
                st.error(f"Could not process transcript: {exc}")
                st.stop()

        st.success("Transcript processed successfully.")

    if "vector_store" in st.session_state:
        video_id = st.session_state.get("video_id", "selected video")
        st.success(f"Transcript ready for {video_id}.")

    question = st.text_input("Ask your question:")

    if st.button("Submit"):
        if not question.strip():
            st.warning("Please enter a question.")
            st.stop()

        if "vector_store" not in st.session_state:
            st.warning("Please collect transcript first.")
            st.stop()

        if not has_huggingface_token():
            st.error("Add HUGGINGFACEHUB_API_TOKEN to your .env file before asking a question.")
            st.stop()

        retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 6})
        docs = retriever.invoke(question)
        prompt = build_prompt(format_docs(docs), question)

        with st.spinner("Thinking..."):
            try:
                answer = generate_answer(prompt)
            except Exception as exc:
                st.error(
                    "Could not generate the answer with Hugging Face. "
                    "Check your token, model, provider availability, or internet connection."
                )
                st.caption(str(exc))
                st.stop()

        st.subheader("Answer:")
        st.write(answer)


if __name__ == "__main__":
    main()
