import os
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime
import uuid

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.summarize import load_summarize_chain
import chromadb

st.set_page_config(page_title="RAG PDF ChatBot", layout="wide")

# UI Styling
st.markdown("""
    <style>
        .stApp { background-color: #1e1e1e; color: white; font-family: 'Segoe UI', sans-serif; }
        .message-container { max-width: 900px; margin: 0 auto; }
        .message { padding: 12px 16px; margin-bottom: 10px; border-radius: 12px; max-width: 85%; word-wrap: break-word; }
        .user-message { background-color: #2b2b2b; margin-left: auto; text-align: right; }
        .bot-message { background-color: #3c3c3c; margin-right: auto; }
        .chat-input { position: fixed; bottom: 10px; width: 100%; max-width: 900px; left: 50%; transform: translateX(-50%); background-color: #121212; padding: 10px; border-radius: 12px; z-index: 100; }
        .block-container { padding-bottom: 100px; }
        .stTextInput > div > div > input { background-color: #1e1e1e; color: white; }
        .stButton > button { background-color: #444; color: white; border: none; }
    </style>
""", unsafe_allow_html=True)

load_dotenv()
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")

# Sidebar Chat Sessions
if 'chat_sessions' not in st.session_state:
    st.session_state.chat_sessions = {}

if 'current_session' not in st.session_state:
    new_session_id = str(uuid.uuid4())
    st.session_state.current_session = new_session_id
    st.session_state.chat_sessions[new_session_id] = {
        "title": "New Chat",
        "messages": [],
        "timestamp": datetime.now()
    }

if st.sidebar.button("➕ New Chat"):
    new_id = str(uuid.uuid4())
    st.session_state.current_session = new_id
    st.session_state.chat_sessions[new_id] = {
        "title": "New Chat",
        "messages": [],
        "timestamp": datetime.now()
    }

st.sidebar.markdown("### 🕒 Chat History")
for sid, chat in sorted(st.session_state.chat_sessions.items(), key=lambda x: x[1]["timestamp"], reverse=True):
    label = f'{chat["title"][:20]} ({chat["timestamp"].strftime("%b %d, %H:%M")})'
    if st.sidebar.button(label, key=sid):
        st.session_state.current_session = sid

current_session_id = st.session_state.current_session

# Load LLM and embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="gemma2-9b-it")

st.markdown("<h2 style='text-align:center;'>📘 Chat with Your PDF</h2>", unsafe_allow_html=True)

if 'store' not in st.session_state:
    st.session_state.store = {}

uploaded_files = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)

if uploaded_files:
    # Detect change in uploaded files
    uploaded_names = [f.name for f in uploaded_files]
    if st.session_state.get('last_uploaded_files') != uploaded_names:
        documents = []
        for uploaded_file in uploaded_files:
            temppdf = "./temp.pdf"
            with open(temppdf, "wb") as f:
                f.write(uploaded_file.getvalue())
            loader = PyPDFLoader(temppdf)
            docs = loader.load()
            documents.extend(docs)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)

        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            client=chroma_client,
            collection_name="rag_pdf_collection"
        )
        retriever = vectorstore.as_retriever()

        # Store in session state
        st.session_state.documents = documents
        st.session_state.retriever = retriever
        st.session_state.last_uploaded_files = uploaded_names

        with st.spinner("Generating summary..."):
            st.subheader("📄 Summary")
            summary_chain = load_summarize_chain(llm, chain_type="map_reduce")
            summary = summary_chain.run(documents[:15])
            st.session_state.summary = summary

        with st.spinner("Generating suggested questions..."):
            st.subheader("💡 Suggested Questions")
            qa_chain = load_qa_chain(llm, chain_type="stuff")
            prompt = "Generate 5-10 questions based on this document:"
            sample_questions = qa_chain.run(input_documents=documents[:5], question=prompt)
            questions = [f"- {q.strip()}" for q in sample_questions.split("\n") if q.strip()]
            st.session_state.suggested_questions = questions[:10]

    # Display summary and questions
    st.subheader("📄 Summary")
    st.markdown(st.session_state.summary)

    st.subheader("💡 Suggested Questions")
    st.markdown("\n".join(st.session_state.suggested_questions))

    st.subheader("💬 Chat Interface")

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question, rewrite it to be standalone if needed."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    history_aware_retriever = create_history_aware_retriever(llm, st.session_state.retriever, contextualize_q_prompt)

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "Use the retrieved context to answer the question concisely.\n\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    def get_session_history(session: str) -> BaseChatMessageHistory:
        if session not in st.session_state.store:
            st.session_state.store[session] = ChatMessageHistory()
        return st.session_state.store[session]

    conversational_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    st.markdown('<div class="message-container">', unsafe_allow_html=True)
    history = get_session_history(current_session_id)
    for msg in history.messages:
        role_class = "user-message" if msg.type == "human" else "bot-message"
        st.markdown(f"<div class='message {role_class}'>{msg.content}</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    with st.container():
        with st.form("chat_form", clear_on_submit=True):
            st.markdown('<div class="chat-input">', unsafe_allow_html=True)
            col1, col2 = st.columns([6, 1])
            user_input = col1.text_input("Ask a question:", key="chat_input", label_visibility="collapsed")
            submit = col2.form_submit_button("➤")
            st.markdown('</div>', unsafe_allow_html=True)

        if submit and user_input:
            st.session_state.chat_sessions[current_session_id]["messages"].append(("user", user_input))
            response = conversational_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": current_session_id}}
            )
            st.session_state.chat_sessions[current_session_id]["messages"].append(("assistant", response['answer']))
            st.rerun()
else:
    st.info("📄 Please upload a PDF to begin.")
