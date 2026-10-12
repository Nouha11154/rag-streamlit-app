import os
import re
import tempfile
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configuration de la page Streamlit
st.set_page_config(page_title="Chat avec mon Livre & Documents", page_icon="🤖", layout="centered")

# --- DESIGN CSS PERSONNALISÉ (COULEURS DES AVATARS) ---
st.markdown("""
    <style>
    /* Changer la couleur de fond des avatars utilisateur (Gris élégant) */
    [data-testid="stChatMessageAvatarUser"] {
        background-color: #4A5568 !important;
    }
    
    /* Changer la couleur de fond des avatars assistant (Bleu professionnel) */
    [data-testid="stChatMessageAvatarAssistant"] {
        background-color: #3182CE !important;
    }

    /* Style des bulles de messages */
    [data-testid="stChatMessage"] {
        border-radius: 10px;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🤖 Assistant IA - RAG Multidocuments")
st.write("Discutez avec vos documents avec correction automatique des fautes et rendu mathématique optimisé !")

# clé API Groq 
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
PERSIST_DIRECTORY = "./chroma_db"

# --- BARRE LATÉRALE : UPLOAD DE FICHIERS ---
st.sidebar.header("📎 Gestion des Documents")
uploaded_file = st.sidebar.file_uploader("Ajouter un fichier PDF ou TXT", type=["pdf", "txt"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    if st.sidebar.button("Indexer ce fichier"):
        with st.spinner("Découpage et indexation du document en cours..."):
            try:
                if uploaded_file.name.endswith(".pdf"):
                    loader = PyPDFLoader(tmp_path)
                else:
                    loader = TextLoader(tmp_path, encoding="utf-8")
                
                docs = loader.load()
                
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1500,
                    chunk_overlap=300,
                    separators=["\n\n", "\n", " ", ""]
                )
                chunks = text_splitter.split_documents(docs)
                
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                vectorstore = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
                vectorstore.add_documents(chunks)
                
                st.sidebar.success(f"✅ Fichier '{uploaded_file.name}' ajouté et indexé avec succès !")
            except Exception as e:
                st.sidebar.error(f"Erreur lors de l'indexation : {e}")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

# --- CHARGEMENT DU RAG ET DU LLM ---
@st.cache_resource
def load_rag_pipeline():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    llm = ChatGroq(model_name="openai/gpt-oss-20b", temperature=0.1)
    
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
        
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain, llm

with st.spinner("Chargement de la base de données..."):
    rag_chain, llm = load_rag_pipeline()

# --- FONCTION DE CORRECTION DES FAUTES DE FRAPPE ---
def correct_user_query(query, llm_model):
    """Corrige automatiquement les fautes d'orthographe de l'utilisateur avant la recherche"""
    correction_prompt = ChatPromptTemplate.from_template(
        "Corrige les fautes d'orthographe ou de frappe dans la question suivante en gardant la même langue et le même sens. Réponds UNIQUEMENT avec la question corrigée, sans aucun texte supplémentaire :\n\n{query}"
    )
    chain = correction_prompt | llm_model | StrOutputParser()
    try:
        corrected = chain.invoke({"query": query})
        return corrected.strip()
    except:
        return query

# --- FONCTION DE FORMATAGE MATHÉMATIQUE ---
def format_math_response(text):
    text = text.replace(r'\(', '$').replace(r'\)', '$')
    text = text.replace(r'\[', '$$').replace(r'\]', '$$')
    return text

# --- HISTORIQUE DE CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Entrée utilisateur
if query := st.chat_input("Posez une question sur vos documents..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Analyse et recherche en cours..."):
            try:
                # 1. Corriger les fautes de frappe de l'utilisateur automatiquement
                clean_query = correct_user_query(query, llm)
                
                # 2. Exécuter le RAG avec la question nettoyée
                response = rag_chain.invoke(clean_query)
                
                # 3. Formater les mathématiques
                final_response = format_math_response(response)
                
                st.markdown(final_response)
                st.session_state.messages.append({"role": "assistant", "content": final_response})
            except Exception as e:
                st.error(f"Une erreur est survenue : {e}")