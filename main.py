import os

# Desactiver explicitement le mode offline pour Hugging Face
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ==========================================
# 1. Configuration du chemin du fichier PDF
# ==========================================
PDF_PATH = "data/book.pdf"

# ==========================================
# 2. Chargement et découpage du document (Chunking)
# ==========================================
print("--> 1. Chargement du fichier PDF...")
loader = PyPDFLoader(PDF_PATH)
documents = loader.load()
print(f"[SUCCESS] Nombre de pages chargées : {len(documents)}")

print("--> 2. Découpage du document en Chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n\n", "\n", " ", ""]
)
chunks = text_splitter.split_documents(documents)
print(f"[SUCCESS] Nombre de chunks créés : {len(chunks)}")

# ==========================================
# 3. Création des Embeddings et Stockage dans ChromaDB
# ==========================================
print("--> 3. Chargement du modèle d'embeddings (HuggingFace)...")

# Modèle léger et très répandu pour générer les vecteurs
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("--> 4. Indexation et sauvegarde dans la base vectorielle ChromaDB...")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print("[SUCCESS] Base vectorielle ChromaDB créée et sauvegardée avec succès !")