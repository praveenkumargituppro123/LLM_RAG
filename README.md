# DocTalk 📜💬

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.30+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-brightgreen.svg)

Welcome to **DocTalk**, a robust Streamlit-based application that empowers users to upload multiple PDF documents, process them, and engage in a ChatGPT-like conversational interface to extract insights. Powered by LangChain, FAISS, and OpenRouter LLM, DocTalk transforms your PDFs into a searchable knowledge base, making it ideal for researchers, students, and professionals.

---

## 🌟 Features

- **📤 Seamless PDF Management**:
  - Upload multiple PDFs via an intuitive sidebar interface.
  - Automatically loads default PDFs from a `./data` directory.
  - Displays all PDFs (default and uploaded) in the sidebar.
  - Triggers processing with a "Confirm" button for efficient resource use.
- **💬 Conversational Chat Interface**:
  - Modern, ChatGPT-like UI with **right-aligned** user queries and **left-aligned** assistant responses.
  - Persists chat history for a continuous conversation.
  - Supports natural language queries about PDF content.
- **📚 Scalable Document Processing**:
  - Processes multiple PDFs using LangChain's `PyPDFDirectoryLoader`.
  - Splits documents into manageable chunks for efficient indexing.
  - Employs FAISS for fast vector search and HuggingFace embeddings for semantic understanding.
- **⚡ Performance Optimizations**:
  - Caches expensive operations with Streamlit's `@st.cache_resource`.
  - Handles unhashable objects to prevent caching errors.
  - Provides visual feedback with spinners and success/error messages.
- **🛡️ Robust Error Handling**:
  - Gracefully handles API failures, missing PDFs, and query errors.
  - Delivers user-friendly feedback for a smooth experience.

---

## 🚀 Getting Started

### Prerequisites
- **Python**: 3.8 or higher
- **OpenRouter API Key**: Obtain one at [openrouter.ai](https://openrouter.ai)
- **Hardware**: 8GB+ RAM recommended for large PDF collections

### Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/doctalk.git
   cd doctalk
   ```

2. **Set Up a Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install streamlit langchain langchain-community pypdf faiss-cpu sentence-transformers requests pydantic python-dotenv
   ```

4. **Configure Environment Variables**:
   - Create a `.env` file in the project root:
     ```plaintext
     OPENROUTER_API_KEY=your-api-key-here
     ```
   - The application loads the API key securely using `python-dotenv`.

5. **Add Default PDFs** (optional):
   - Place PDF files in the `./data` directory to load them by default.

### Running the Application

```bash
streamlit run app.py
```

The app will open in your browser (default: `http://localhost:8501`).

---

## 🖥️ Usage

### 1. **Manage PDFs**:
   - In the sidebar, view default PDFs from `./data` and upload additional PDFs.
   - Uploaded PDFs are saved to `./Uploads` and listed in the sidebar.
   - Click the **Confirm** button to start processing.

### 2. **Chat with Your PDFs**:
   - After processing, use the chat input at the bottom to ask questions about the PDFs.
   - Your queries appear **right-aligned**, and assistant responses are **left-aligned**, styled like a modern chatbot.
   - Example: "What is the main topic of the uploaded PDFs?"

### 3. **Monitor Progress**:
   - A simple "Processing PDFs..." message appears during processing.
   - Once complete, a success message indicates the app is ready for chatting.

---

## 📝 How the Chatbot Works

### **Process Behind the Scenes**:
1. **PDF Loading and Splitting**:
   - PDFs are loaded from `./Uploads` using `PyPDFDirectoryLoader`.
   - Documents are split into smaller chunks (default: 1000 characters) with overlap for better context retention.

2. **Embedding and Indexing**:
   - Text chunks are embedded using HuggingFace's `all-MiniLM-L6-v2` model.
   - FAISS creates a vector store for fast similarity search.

3. **LLM Integration**:
   - OpenRouter LLM (default: `mistralai/devstral-small:free`) powers the chatbot.
   - The RetrievalQA chain combines the vector store and LLM for accurate responses.

4. **Chat Interface**:
   - User queries are processed using the RetrievalQA chain.
   - Responses are generated based on the most relevant PDF chunks.
   - Chat history is maintained for a continuous conversation.

---

## ⚙️ Configuration

- **API Key**: Store your OpenRouter API key in `.env` for secure access.
- **Embedding Model**: Default is `all-MiniLM-L6-v2`. For higher accuracy, update `model_name` in `app.py` to `all-mpnet-base-v2` (requires more resources).
- **Document Chunking**: Adjust `chunk_size` (default: 1000) and `chunk_overlap` (default: 200) in `app.py` for large PDFs.
- **Chat Styling**: Customize the CSS in `app.py` (within `st.markdown`) to change message colors, fonts, or bubble shapes.

---

## 📁 Project Structure

```plaintext
doctalk/
├── app.py              # Main Streamlit application
├── data/               # Directory for default PDFs (optional)
├── Uploads/            # Directory for uploaded PDFs
├── .env                # Environment variables (API key)
├── README.md           # Project documentation
└── requirements.txt    # Python dependencies
```

Generate a `requirements.txt`:
```bash
pip freeze > requirements.txt
```

---

## 🛠️ Performance Optimizations

- **Caching**: Leverages `@st.cache_resource` for document loading, splitting, embeddings, and vector store creation.
- **Error Prevention**: Uses underscore-prefixed parameters to handle unhashable objects, avoiding Streamlit's `UnhashableParamError`.
- **Lazy Processing**: Delays processing until the "Confirm" button is clicked.
- **Efficient Search**: FAISS enables fast similarity search for large PDF collections.

---

## 🤝 Contributing

We welcome contributions to make DocTalk even better! To contribute:
1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a pull request.

Please include tests and update documentation as needed.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 📬 Contact

For questions, feedback, or support, please:
- Open an issue on the [GitHub repository](https://github.com/your-username/doctalk).
- Email: [praveenmono711@gmail.com](mailto:praveenmono711@gmail.com)

---

**DocTalk**: Your intelligent companion for exploring PDF content! 🚀