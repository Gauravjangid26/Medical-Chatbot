from langchain.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import GoogleGenerativeAIEmbeddings
import os

# Load files
loader = DirectoryLoader("data", glob="**/*.txt", loader_cls=TextLoader)
docs = loader.load()

# Split
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(docs)

# Embedding
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Store vector
vectorstore = FAISS.from_documents(chunks, embedding)
vectorstore.save_local("vectorstore")
