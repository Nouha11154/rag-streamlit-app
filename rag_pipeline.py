import os

import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Mettez votre clé API Groq ici
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

print("--> 1. Chargement de la base ChromaDB et des embeddings...")
PERSIST_DIRECTORY = "./chroma_db"
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

print("--> 2. Initialisation du modèle Groq...")
llm = ChatGroq(
    model_name="openai/gpt-oss-20b",
    temperature=0.1
)

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

print("\n==================================================")
print("🤖 VOTRE CHATBOT 'AI AGENTS IN ACION' EST PRÊT !")
print("Tapez votre question ou écrivez 'exit' / 'quitter' pour fermer.")
print("==================================================\n")

# Boucle interactive pour poser plusieurs questions
while True:
    query = input("💬 Votre question : ")
    
    # Condition pour quitter le programme
    if query.strip().lower() in ["exit", "quitter", "quit"]:
        print("Au revoir !")
        break
    
    if not query.strip():
        continue

    print("\n⏳ Recherche et génération en cours...\n")
    try:
        response = rag_chain.invoke(query)
        print("---------------- RÉPONSE ----------------")
        print(response)
        print("-----------------------------------------\n")
    except Exception as e:
        print(f"❌ Erreur : {e}\n")