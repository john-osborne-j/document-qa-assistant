import streamlit as st
import os
import tempfile
import dotenv
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# Load environment variables
dotenv.load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'not-needed')
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY', '')
os.environ['LANGCHAIN_TRACING_V2'] = 'True'
os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT', 'streamlit-qa-app')
# Page config
st.set_page_config(
    page_title="Document Q&A Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        color: white;
        text-align: center;
        margin: 0;
    }
    .main-header p {
        color: white;
        text-align: center;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    .stAlert > div {
        padding: 1rem;
        border-radius: 10px;
    }
    .upload-box {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: #f8f9fa;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: #333;
    }
    .user-message {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
        color: #1565c0;
    }
    .assistant-message {
        background: #f1f8e9;
        border-left: 4px solid #4caf50;
        color: #2e7d32;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'document_processed' not in st.session_state:
    st.session_state.document_processed = False

def process_document(uploaded_file):
    """Process uploaded PDF and create vector store"""
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        # Data Ingestion
        with st.spinner("Loading PDF..."):
            loader = PyPDFLoader(tmp_file_path)
            docs = loader.load()
            
        if not docs:
            st.error("No content found in the PDF. Please check the file.")
            return False
            
        # Data Transformation
        with st.spinner("Splitting document into chunks..."):
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = splitter.split_documents(docs)
            
        # Create embeddings and vector store
        with st.spinner("Creating embeddings (this may take a moment)..."):
            embeddings = OllamaEmbeddings(model='llama3.2')
            vectorstore = FAISS.from_documents(splits, embedding=embeddings)
            
        # Store in session state
        st.session_state.vectorstore = vectorstore
        st.session_state.document_processed = True
        
        # Create QA chain
        retriever = vectorstore.as_retriever()
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an AI assistant that answers questions based on the provided context from a PDF document. 
            Use only the information from the context to answer questions accurately and concisely. 
            If the answer is not in the context, clearly state that the information is not available in the document.
            
            Context: {context}"""),
            ("user", "{question}")
        ])
        
        llm = Ollama(model='llama3.2')
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        qa_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        st.session_state.qa_chain = qa_chain
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        st.success(f"✅ Document processed successfully! Found {len(splits)} text chunks.")
        return True
        
    except Exception as e:
        st.error(f"❌ Error processing document: {str(e)}")
        return False

def get_answer(question):
    """Get answer from the QA chain"""
    try:
        if st.session_state.qa_chain is None:
            return "Please upload and process a document first."
            
        with st.spinner("Thinking..."):
            response = st.session_state.qa_chain.invoke(question)
            return response
            
    except Exception as e:
        return f"Error generating answer: {str(e)}"

# Main app layout
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📄 Document Q&A Assistant</h1>
        <p>Upload a PDF document and ask questions about its content using AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🔧 Settings")
        
        # Clear session button
        if st.button("🗑️ Clear Session", type="secondary"):
            st.session_state.vectorstore = None
            st.session_state.qa_chain = None
            st.session_state.chat_history = []
            st.session_state.document_processed = False
            st.success("Session cleared!")
            st.rerun()
        
        st.markdown("---")
        
        # Document info
        if st.session_state.document_processed:
            st.markdown("### 📋 Document Status")
            st.success("✅ Document loaded and ready!")
        else:
            st.markdown("### 📋 Document Status")
            st.warning("⏳ No document loaded")
        
        st.markdown("---")
        
        # Instructions
        st.markdown("""
        ### 📖 How to use:
        1. Upload a PDF document
        2. Wait for processing to complete
        3. Ask questions about the content
        4. Get AI-powered answers!
        
        ### 💡 Tips:
        - Ask specific questions for better results
        - The AI can only answer based on document content
        - Try different phrasings if needed
        """)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Upload Document")
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload a PDF document to analyze"
        )
        
        if uploaded_file is not None:
            st.info(f"📄 File uploaded: {uploaded_file.name}")
            
            if st.button("🚀 Process Document", type="primary"):
                success = process_document(uploaded_file)
                if success:
                    st.rerun()
    
    with col2:
        st.markdown("### ❓ Ask Questions")
        
        # Question input
        question = st.text_input(
            "Enter your question:",
            placeholder="What is this document about?",
            disabled=not st.session_state.document_processed,
            help="Ask questions about the uploaded document"
        )
        
        # Submit button
        if st.button("🎯 Get Answer", type="primary", disabled=not st.session_state.document_processed):
            if question.strip():
                # Add question to chat history
                st.session_state.chat_history.append({"role": "user", "content": question})
                
                # Get answer
                answer = get_answer(question)
                
                # Add answer to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                
                # Clear input
                st.rerun()
            else:
                st.warning("Please enter a question!")
    
    # Chat history
    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("### 💬 Chat History")
        
        # Display chat messages using Streamlit's built-in chat elements
        for i, message in enumerate(st.session_state.chat_history):
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(message["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(message["content"])
        
        # Clear chat history button
        if st.button("🧹 Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

# Check for required dependencies
def check_dependencies():
    """Check if Ollama is available"""
    try:
        # Test if Ollama is running
        test_llm = Ollama(model='llama3.2')
        return True
    except Exception as e:
        st.error(f"""
        ❌ **Ollama Setup Required**
        
        Please ensure Ollama is installed and running with the llama3.2 model:
        
        1. Install Ollama from https://ollama.ai
        2. Run: `ollama pull llama3.2`
        3. Make sure Ollama service is running
        
        Error: {str(e)}
        """)
        return False

if __name__ == "__main__":
    # Check dependencies first
    if check_dependencies():
        main()
    else:
        st.stop()
