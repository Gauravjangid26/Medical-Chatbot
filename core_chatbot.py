from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os

load_dotenv()

embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = FAISS.load_local("vectorstore", embedding, allow_dangerous_deserialization=True)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", temperature=0)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory
)

def ask_bot(query: str):
    return qa_chain.run(query)
