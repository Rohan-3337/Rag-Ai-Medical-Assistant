import streamlit as st
import os
from dotenv import load_dotenv
import tempfile
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Set API keys
os.environ["GROQ_API_KEY"] = "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"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"

# Page configuration
st.set_page_config(
    page_title="Medical Report Assistant",
    page_icon="🩺",
    layout="wide"
)

if 'retrieval_chain' not in st.session_state:
    st.session_state.retrieval_chain = None
if 'file_processed' not in st.session_state:
    st.session_state.file_processed = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

st.sidebar.title("Upload Document")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

model_option = st.sidebar.selectbox(
    "Select LLM Model",
    ["llama-3.1-8b-instant"],
    index=0
)

temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)

def process_document(uploaded_file):
    with st.spinner("Processing document..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            loader = PyPDFLoader(tmp_file_path)
            docs = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            documents = text_splitter.split_documents(docs)
            
            st.sidebar.success(f"Document processed: {len(documents)} chunks created")
            
            embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
            
            db = FAISS.from_documents(documents, embeddings)
            retriever = db.as_retriever(search_kwargs={"k": 3})
            
            llm = ChatGroq(model=model_option, temperature=temperature)
            
            prompt = ChatPromptTemplate.from_template("""
            Answer the following question based only on the provided context.
            Think step by step before providing a detailed answer.
            You are helping to simplify medical reports and help users understand them.
            If you don't know the answer based on the context, clearly state that you don't know.
            
            <context>
            {context}
            </context>
            
            Question: {input}
            """)
            
            document_chain = create_stuff_documents_chain(llm, prompt)
            
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            
            st.session_state.retrieval_chain = retrieval_chain
            st.session_state.file_processed = True
            
            os.unlink(tmp_file_path)
            
            return True
            
        except Exception as e:
            st.error(f"Error processing document: {str(e)}")
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
            return False

st.title("Medical Report Assistant 🩺")
st.write("Upload a medical report and ask questions to understand it better.")

if uploaded_file and not st.session_state.file_processed:
    process_document(uploaded_file)
elif uploaded_file and st.sidebar.button("Reprocess Document"):
    st.session_state.file_processed = False
    st.session_state.chat_history = []
    process_document(uploaded_file)

if st.session_state.file_processed:
    for message in st.session_state.chat_history:
        role = message["role"]
        content = message["content"]
        
        if role == "user":
            st.chat_message(role, avatar="🧑").write(content)
        else:
            st.chat_message(role, avatar="🤖").write(content)
  
    user_query = st.chat_input("Ask a question about your medical report...")
    
    if user_query:

        st.chat_message("user", avatar="🧑").write(user_query)
        
   
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        with st.spinner("Thinking..."):
            response = st.session_state.retrieval_chain.invoke({"input": user_query})
            answer = response['answer']

        st.chat_message("assistant", avatar="🤖").write(answer)

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
else:
    if not uploaded_file:
        st.info("Please upload a PDF document to get started.")
    else:
        st.warning("Document processing failed. Please try again with a different file.")

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "This app helps simplify medical reports using RAG (Retrieval Augmented Generation) technology. "
    "Upload your medical report and ask questions in plain language to better understand your health information."
)