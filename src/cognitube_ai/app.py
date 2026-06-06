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



# -------------------- MAIN --------------------
def main():

    st.set_page_config(page_title="YouTube RAG Chatbot", layout="centered")
    st.title("🤖 YouTube RAG Chatbot")

#-------------------Video Transcript Generate-----------------------
    video_input = st.text_input("Enter YouTube URL or Video ID:")

    if st.button('Collect Transcript'):

        if not video_input:
            st.warning("Please enter Video URL.")
            st.stop()

        video_id = extract_video_id(video_input)

        with st.spinner("📥 Fetching transcript..."):
            try:
                transcript = get_transcript(video_id)
            except Exception as e:
                st.error(f'Error fetching transcript: {e}')
                st.stop()

        if transcript:
            try:
                st.session_state.vector_store = build_vector_store(transcript)
                st.success("✅ Transcript processed successfully!")
            except Exception as e:
                st.error(f"Error building vector store: {e}")

        if "vector_store" in st.session_state:
            st.success("📚 Transcript Ready")

#--------------User Query---------------------
    question = st.text_input("Ask your question:")

    if st.button("Submit"):

        if not question.strip():
            st.warning('⚠️ Please enter a question.')
            st.stop()

        if 'vector_store' not in st.session_state:
            st.warning('⚠️ Please collect transcript first.')
            st.stop()

        retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 6})

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

        if 'llm' not in st.session_state:
            st.session_state.llm = get_llm()

        rag_chain = parallel_chain | prompt | st.session_state.llm | parser

        with st.spinner("🤖 Thinking..."):
            try:
                answer = rag_chain.invoke(question)
                st.subheader("🤖 Answer:")
                st.write(answer)

            except Exception as e:
                st.error(f'Error generating answer: {e}')



if __name__ == "__main__":
    main()
