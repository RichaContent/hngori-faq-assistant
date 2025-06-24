import os
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# Load API key from Streamlit secrets
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

st.set_page_config(page_title="🧘 Hingori FAQ Assistant")
st.title("🧘 Hingori Spiritual FAQ Assistant")

@st.cache_resource
def load_documents():
    docs = []
    for file in os.listdir("data"):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(f"data/{file}")
            docs.extend(loader.load())
    return docs

@st.cache_resource
def create_qa_chain():
    raw_docs = load_documents()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(raw_docs)
    vectorstore = FAISS.from_documents(chunks, OpenAIEmbeddings())
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    llm = OpenAI(temperature=0)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

qa_chain = create_qa_chain()

question = st.text_input("Ask your spiritual question below:")

if question:
    with st.spinner("Reflecting deeply..."):
        result = qa_chain({"query": question})
        st.markdown("### 🌟 Answer")
        st.write(result["result"])

        st.markdown("### 📚 Sources")
        for doc in result["source_documents"]:
            st.write(f"• {doc.metadata.get('source', 'Unknown')} (p. {doc.metadata.get('page', '?')})")
