import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ==========================================
# 1. Configuration de la base de données existante
# ==========================================
PERSIST_DIRECTORY = "./chroma_db"

print("--> 1. Chargement du modèle d'embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("--> 2. Connexion à la base vectorielle ChromaDB...")
# On recharge la base de données existante sans tout recalculer
vectorstore = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embeddings
)

# ==========================================
# 2. Poser une question au système (Retrieval)
# ==========================================
query = "What is an AI Agent?"  # يمكنك تغيير هذا السؤال حسب رغبتك
print(f"\n--> 3. Recherche des passages pertinents pour la question : '{query}'\n")

# Recherche des 3 chunks les plus proches/pertinents
docs = vectorstore.similarity_search(query, k=3)

# Affichage des résultats trouvés
for i, doc in enumerate(docs, 1):
    print(f"--- Résultat {i} ---")
    print(doc.page_content[:400] + "...\n")  # Afficher un extrait du chunk
    print(f"(Source : Page {doc.metadata.get('page', 'Inconnue')})\n")