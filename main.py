from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

app = FastAPI()

# Enable CORS (Important for frontend apps to access the API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load dataset
with open("disease_data.json", "r") as f:
    diseases = json.load(f)

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")
symptom_texts = [d["symptoms"] for d in diseases]
symptom_embeddings = model.encode(symptom_texts)

# Store embeddings in FAISS
index = faiss.IndexFlatL2(symptom_embeddings.shape[1])
index.add(np.array(symptom_embeddings))

@app.get("/chat")
def chat(symptom_query: str):
    query_embedding = model.encode([symptom_query])
    _, idx = index.search(np.array(query_embedding), 1)

    disease = diseases[idx[0][0]]
    return {
        "disease": disease["disease"],
        "cause": disease["cause"],
        "cure": disease["cure"]
    }

# Run with Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
