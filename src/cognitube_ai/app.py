import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough

from cognitube_ai.backend import (
    build_vector_store,
    extract_video_id,
    get_llm,
    get_transcript,
    has_huggingface_token,
    parser,
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def main():
    st.set_page_config(page_title="YouTube RAG Chatbot", layout="centered")
    st.title("YouTube RAG Chatbot")

    video_input = st.text_input("Enter YouTube URL or Video ID:")
    question = st.text_input("Ask your question:")

    if not st.button("Submit"):
        return

    if not video_input or not question:
        st.warning("Please enter both Video ID/URL and Question.")
        st.stop()

    if not has_huggingface_token():
        st.error("Add HUGGINGFACEHUB_API_TOKEN to your .env file before running the app.")
        st.stop()

    video_id = extract_video_id(video_input)

    with st.spinner("Fetching transcript..."):
        transcript = get_transcript(video_id)

    if not transcript:
        st.stop()

    with st.spinner("Processing transcript..."):
        vector_store = build_vector_store(transcript)

    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    prompt = PromptTemplate(
        template="""
You are a helpful assistant.
Answer ONLY from the provided transcript context.
If the context is insufficient, say "I don't know."

Context:
{context}

Question:
{question}

Answer:
""",
        input_variables=["context", "question"],
    )

    parallel_chain = RunnableParallel(
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
    )

    rag_chain = parallel_chain | prompt | get_llm() | parser

    with st.spinner("Thinking..."):
        answer = rag_chain.invoke(question)

    st.subheader("Answer:")
    st.write(answer)


if __name__ == "__main__":
    main()
