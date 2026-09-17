"""Hybrid, cited document RAG using NVIDIA NIM."""

from __future__ import annotations

import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_core import Chunk, lexical_rank, reciprocal_rank_fusion

load_dotenv()
st.set_page_config(page_title="NVIDIA NIM Knowledge Assistant", layout="wide")
st.title("NVIDIA NIM Knowledge Assistant")
st.caption("Hybrid dense + lexical retrieval with page-level citations")

api_key = st.sidebar.text_input(
    "NVIDIA API key", type="password", value=os.getenv("NVIDIA_API_KEY", "")
)
top_k = st.sidebar.slider("Retrieved passages", 2, 8, 4)
uploads = st.file_uploader("Upload PDF knowledge sources", type="pdf", accept_multiple_files=True)

if "rag" not in st.session_state:
    st.session_state.rag = None

if st.button("Build knowledge index", disabled=not uploads or not api_key):
    os.environ["NVIDIA_API_KEY"] = api_key
    documents = []
    for upload in uploads:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as handle:
            handle.write(upload.getvalue())
            path = handle.name
        loaded = PyPDFLoader(path).load()
        for document in loaded:
            document.metadata["source"] = upload.name
        documents.extend(loaded)
        os.unlink(path)

    splits = RecursiveCharacterTextSplitter(
        chunk_size=900, chunk_overlap=140
    ).split_documents(documents)
    chunks = [
        Chunk(
            chunk_id=index,
            text=document.page_content,
            source=document.metadata.get("source", "document"),
            page=int(document.metadata.get("page", 0)) + 1,
        )
        for index, document in enumerate(splits)
    ]
    embeddings = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5")
    vectorstore = FAISS.from_texts(
        [chunk.text for chunk in chunks],
        embeddings,
        metadatas=[{"chunk_id": chunk.chunk_id} for chunk in chunks],
    )
    st.session_state.rag = (chunks, vectorstore)
    st.success(f"Indexed {len(chunks)} passages from {len(uploads)} PDF(s).")

question = st.chat_input("Ask a question about the indexed documents")

if question:
    if not st.session_state.rag:
        st.error("Build the knowledge index first.")
        st.stop()

    os.environ["NVIDIA_API_KEY"] = api_key
    chunks, vectorstore = st.session_state.rag
    dense_docs = vectorstore.similarity_search(question, k=max(top_k * 2, 8))
    dense_ids = [int(doc.metadata["chunk_id"]) for doc in dense_docs]
    lexical_ids = lexical_rank(question, chunks)[: max(top_k * 2, 8)]
    fused_ids = reciprocal_rank_fusion(dense_ids, lexical_ids)[:top_k]
    selected = [chunks[index] for index in fused_ids]

    context = "\n\n".join(
        f"[{i}] {chunk.source}, page {chunk.page}\n{chunk.text}"
        for i, chunk in enumerate(selected, start=1)
    )
    llm = ChatNVIDIA(model="meta/llama-3.1-70b-instruct", temperature=0)
    response = llm.invoke(
        f"""Answer only from the supplied passages.
Cite factual statements with passage numbers such as [1].
If the passages do not support an answer, say so.

Question: {question}

Passages:
{context}"""
    ).content

    st.chat_message("user").write(question)
    st.chat_message("assistant").write(response)
    with st.expander("Retrieved evidence"):
        for index, chunk in enumerate(selected, start=1):
            st.markdown(f"**[{index}] {chunk.source}, page {chunk.page}**")
            st.write(chunk.text)
