
# 📄 Document Q&A Assistant

A powerful AI-powered web application that allows users to upload PDF documents and ask questions about their content using natural language processing and local AI models.


##  Features

- PDF Document Upload: Easy drag-and-drop PDF file upload
- AI-Powered Q&A: Ask natural language questions about document content
- Local AI Processing: Uses Ollama with Llama 3.2 for privacy and offline capability
- Intelligent Document Chunking: Splits documents into manageable chunks for better processing
- Vector Search: FAISS-based similarity search for relevant content retrieval
- Interactive Chat Interface: Clean, user-friendly chat history display
- Session Management: Clear sessions and chat history as needed
- Responsive Design: Modern UI with gradient styling and responsive layout

## Demo
![Image](https://github.com/user-attachments/assets/eefc708d-9b3e-4cea-b8b1-7dfafe88d9e6)

![Image](https://github.com/user-attachments/assets/233e2447-d08a-48e4-a468-00409887d352)

![Image](https://github.com/user-attachments/assets/0b2190f4-436e-4460-b15b-6474249695a2)

![Image](https://github.com/user-attachments/assets/5871b09e-b7b5-4601-8e2a-8fee0816bd3e)

![Image](https://github.com/user-attachments/assets/7b8d1444-e7bc-438a-81cc-77c883bb1e5e)


##  Prerequisites

Before running this application, ensure you have:

- #### Python 3.8+ installed
- #### Ollama installed and running locally
- #### Llama 3.2 model pulled in Ollama
## Setting up Ollama  

- ## Install Ollama 
  ```from https://ollama.ai ```
- ## Pull the required model: 

```bash
   ollama pull llama3.2
```
- Ensure Ollama service is running (it typically starts automatically)


##  Installation

### Clone the repository:
```bash
git clone https://github.com/john-osborne-j/document-qa-assistant.git
```
```
cd document-qa-assistant
```

### Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install dependencies:
```
pip install -r requirements.txt
```

### Set up environment variables (optional):
- #### Create a .env file in the project root:
```
OPENAI_API_KEY=not-needed
LANGCHAIN_API_KEY=your_langchain_api_key_here
LANGCHAIN_PROJECT=streamlit-qa-app
```
    
##  Running the Application
Start the Streamlit app:
```
streamlit run app.py
```

- Open your browser and navigate to http://localhost:8501
- Upload a PDF and start asking questions!
##  How It Works
-  Document Processing: PDFs are loaded using PyPDFLoader and split into chunks using RecursiveCharacterTextSplitter
- Embeddings Generation: Text chunks are converted to embeddings using Ollama embeddings
- Vector Storage: Embeddings are stored in a FAISS vector database for efficient similarity search
- Question Processing: User questions are processed through a RAG (Retrieval-Augmented Generation) pipeline
- Answer Generation: Relevant document chunks are retrieved and used as context for the Llama 3.2 model to generate answers
##  Privacy & Security

- Local Processing: All AI processing happens locally using Ollama - no data sent to external APIs
- Temporary Files: Uploaded PDFs are temporarily stored and automatically cleaned up
- Session-Based: Document data is stored only in the current session
##  Configuration

### Environment Variables

- OPENAI_API_KEY: Not needed for local Ollama usage
- LANGCHAIN_API_KEY: Optional, for LangSmith tracing
- LANGCHAIN_TRACING_V2: Enable LangSmith tracing
- LANGCHAIN_PROJECT: Project name for LangSmith

## Model Configuration
- You can change the Ollama model by modifying these lines in app.
```bash
pythonembeddings = OllamaEmbeddings(model='llama3.2')  # Change embedding model
llm = Ollama(model='llama3.2')  # Change LLM model
```

##  Customization

- ### Chunk Size Configuration
```
pythonsplitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,    # Adjust chunk size
    chunk_overlap=200   # Adjust overlap
)
```
##  Performance Tips

### Large Documents: 
- For very large PDFs, consider increasing chunk size or implementing pagination
### Multiple Documents: 
- Extend the app to handle multiple PDFs simultaneously
### Caching: 
- Implement Streamlit caching for better performance with repeated queries
##  Contributing 
-  Fork the repository Create a feature branch 
```(git checkout -b feature/amazing-feature)```
- Commit your changes 
```(git commit -m 'Add amazing feature') ```
- Push to the branch ```(git push origin feature/amazing-feature)```
- Open a Pull Request
## 🙏 Acknowledgments  
- Streamlit for the amazing web framework 
- LangChain for the AI/ML toolkit Ollama for
- local AI model serving FAISS for efficient similarity search
## 📞 Support
- if you encounter any issues or have questions:  - Check the Troubleshooting Guide 
- Open an issue on GitHub
- Contact [osbornej406@gmail.com]

-------------------------------------------------------------
  ⭐ Star this repository if you found it helpful! ⭐
