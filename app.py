import os
import streamlit as st
from langchain.llms import BaseLLM
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.schema import LLMResult, Generation
from pydantic import BaseModel, Field
import requests
import json
import shutil
from dotenv import load_dotenv

# Function to create or update .env file with API key
def update_env_file(api_key: str):
    env_file = ".env"
    env_content = f"OPENROUTER_API_KEY={api_key}\n"
    try:
        with open(env_file, "w") as f:
            f.write(env_content)
        load_dotenv(override=True)
        return True
    except Exception as e:
        st.error(f"Error updating .env file: {str(e)}")
        return False

# Function to update data_sources.txt with uploaded PDF names
def update_data_sources(upload_dir):
    data_sources_file = os.path.join(upload_dir, "data_sources.txt")
    pdf_files = [f for f in os.listdir(upload_dir) if f.endswith(".pdf")]
    with open(data_sources_file, "w") as f:
        for pdf in pdf_files:
            f.write(f"{pdf}\n")

# Load environment variables from .env file
load_dotenv()

# Custom OpenRouter LLM
class OpenRouterLLM(BaseLLM, BaseModel):
    api_key: str = Field(..., description="OpenRouter API key")
    model: str = Field(default="mistralai/devstral-small:free", description="Model name for OpenRouter API")
    url: str = Field(default="https://openrouter.ai/api/v1/chat/completions", description="OpenRouter API endpoint")

    class Config:
        arbitrary_types_allowed = True

    def _call(self, prompt: str, stop=None) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "",  # Replace with your site URL
            "X-Title": "",  # Replace with your site title
        }
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}]
        }
        if stop:
            data["stop"] = stop
        try:
            response = requests.post(self.url, headers=headers, data=json.dumps(data))
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except requests.RequestException as e:
            st.error(f"Error communicating with OpenRouter API: {str(e)}")
            return ""

    def _generate(self, prompts: list, stop=None) -> LLMResult:
        generations = []
        for prompt in prompts:
            text = self._call(prompt, stop=stop)
            generations.append([Generation(text=text)])
        return LLMResult(generations=generations)

    @property
    def _llm_type(self) -> str:
        return "openrouter"

# Load PDFs
@st.cache_resource
def load_data(directory: str):
    if not os.path.exists(directory):
        os.makedirs(directory)  # Create directory if it doesn't exist
    loader = PyPDFDirectoryLoader(directory)
    data = loader.load()
    if not data:
        st.warning("No documents loaded. Please upload PDF files or ensure the directory contains PDFs.")
    return data

# Split documents
@st.cache_resource
def split_documents(_data: list):
    if not _data:
        return []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(_data)

# Initialize embeddings
@st.cache_resource
def initialize_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create FAISS vector store
@st.cache_resource
def create_vector_store(_texts: list, _embeddings):
    if not _texts:
        st.error("No text chunks available to create vector store.")
        return None
    return FAISS.from_documents(_texts, _embeddings)

# Initialize LLM
@st.cache_resource
def initialize_llm(_api_key: str):
    return OpenRouterLLM(api_key=_api_key)

# Set up RetrievalQA chain
@st.cache_resource
def setup_qa_chain(_llm, _vectorstore):
    if _vectorstore is None:
        return None
    return RetrievalQA.from_chain_type(
        llm=_llm,
        chain_type="stuff",
        retriever=_vectorstore.as_retriever(),
        return_source_documents=False  # Disable source documents
    )

# Streamlit UI
def main():
    st.title("PDF Chatbot")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "upload_dir" not in st.session_state:
        st.session_state.upload_dir = "./Uploads"
    if "data_copied" not in st.session_state:
        st.session_state.data_copied = False
    if "processing_triggered" not in st.session_state:
        st.session_state.processing_triggered = False
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None

    # Sidebar for file upload, management, and settings
    with st.sidebar:
        st.header("PDF Management")
        
        # Display default and uploaded PDFs
        st.subheader("Available PDFs")
        os.makedirs(st.session_state.upload_dir, exist_ok=True)
        
        # Copy default PDFs from ./data to ./Uploads if not already done
        if not st.session_state.data_copied and os.path.exists("./data"):
            for file in os.listdir("./data"):
                if file.endswith(".pdf"):
                    shutil.copy(os.path.join("./data", file), os.path.join(st.session_state.upload_dir, file))
            st.session_state.data_copied = True

        # List all PDFs in the upload directory
        pdf_files = [f for f in os.listdir(st.session_state.upload_dir) if f.endswith(".pdf")]
        if pdf_files:
            for pdf in pdf_files:
                st.write(f"- {pdf}")
        else:
            st.write("No PDFs available. Upload files below.")

        # File uploader
        uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
        if uploaded_files:
            with st.spinner("Saving uploaded PDFs..."):
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(st.session_state.upload_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                st.success(f"Uploaded {len(uploaded_files)} PDF(s).")
                # Update data_sources.txt with new PDF names
                update_data_sources(st.session_state.upload_dir)
                # Refresh file list
                pdf_files = [f for f in os.listdir(st.session_state.upload_dir) if f.endswith(".pdf")]

        # Confirm button to trigger processing
        if st.button("Confirm"):
            if pdf_files:
                st.session_state.processing_triggered = True
                st.success("Processing started...")
            else:
                st.error("No PDFs available to process. Please upload at least one PDF.")

        # Settings section
        st.header("Settings")
        st.subheader("OpenRouter API Key")
        current_api_key = os.getenv("OPENROUTER_API_KEY")
        new_api_key = st.text_input("Enter new API key", type="password", value=current_api_key if current_api_key else "")
        if st.button("Save API Key"):
            if new_api_key:
                if update_env_file(new_api_key):
                    st.success("API key updated successfully. Please click 'Confirm' to reprocess with the new key.")
                    st.session_state.processing_triggered = False
                    st.session_state.qa_chain = None
                else:
                    st.error("Failed to update API key.")
            else:
                st.error("Please enter a valid API key.")

    # Process PDFs only if Confirm is clicked and PDFs are available
    if st.session_state.processing_triggered and pdf_files:
        with st.spinner("Loading documents..."):
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            data = load_data(st.session_state.upload_dir)
        st.write(f"Number of documents loaded: {len(data)}")

        if not data:
            st.session_state.processing_triggered = False
            st.stop()

        with st.spinner("Splitting documents..."):
            texts = split_documents(data)
        st.write(f"Number of text chunks: {len(texts)}")

        if not texts:
            st.session_state.processing_triggered = False
            st.stop()

        with st.spinner("Initializing embeddings..."):
            embeddings = initialize_embeddings()

        with st.spinner("Creating vector store..."):
            vectorstore = create_vector_store(texts, embeddings)
        if vectorstore is None:
            st.session_state.processing_triggered = False
            st.stop()

        st.success("PDF processing complete. Ready to chat!")

        # Initialize LLM
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            st.error("OpenRouter API key not found. Please create a .env file with OPENROUTER_API_KEY or update it in Settings.")
            st.session_state.processing_triggered = False
            st.stop()
        with st.spinner("Initializing LLM..."):
            llm = initialize_llm(api_key)

        # Set up RetrievalQA chain
        with st.spinner("Setting up QA chain..."):
            st.session_state.qa_chain = setup_qa_chain(llm, vectorstore)
        if st.session_state.qa_chain is None:
            st.session_state.processing_triggered = False
            st.stop()

    # Chat interface
    if st.session_state.qa_chain:
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # User input
        if prompt := st.chat_input("Ask a question about the PDFs"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        result = st.session_state.qa_chain({"query": prompt})
                        response = result["result"]
                        st.markdown(response)
                        # Add assistant response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error(f"Error processing query: {str(e)}")
    else:
        st.info("Please upload PDFs and click 'Confirm' to start processing.")

if __name__ == "__main__":
    main()