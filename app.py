import os
import tempfile
import streamlit as st
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader

st.set_page_config(page_title="Textbook AI")
st.title("Textbook AI")

llm = ChatOllama(model="llama3.2", temperature=0.2)
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
persist_dir = "./chroma_db"

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    with st.spinner("Processing PDF..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )
        chunks = splitter.split_documents(documents)

        Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_dir
        )

        st.success("PDF indexed successfully!")

if os.path.exists(persist_dir):
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Answer only using the provided context. "
            "If the answer is not found, say 'I don't know'."
        ),
        (
            "human",
            "Context:\n{context}\n\nQuestion:\n{question}"
        ),
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    query = st.text_input("Ask a question about the PDF")

    if query:
        with st.spinner("Thinking..."):
            response = rag_chain.invoke(query)

        st.subheader("Answer")
        st.write(response)