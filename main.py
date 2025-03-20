import os
import pickle
import gdown
import numpy as np
from flask import Flask, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Google Drive file ID for the model
FILE_ID = "1dEqytupX4VkZVglj1EyVZvGmenBvKzf2"  # Replace with your actual Google Drive file ID
MODEL_PATH = "disease_model.pkl"

# Download the model from Google Drive if not present
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH, quiet=False)

# Load the pre-trained model and embeddings from the .pkl file
with open(MODEL_PATH, "rb") as f:
    data = pickle.load(f)

diseases = data["diseases"]
model = data["model"]  # ✅ Model is already inside the pickle file!
symptom_embeddings = data["embeddings"]  # ✅ Precomputed symptom embeddings

@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_input = request.json["message"]
        user_embedding = model.encode([user_input])  # ✅ Uses preloaded model

        # Compute similarity with precomputed symptom embeddings
        similarities = cosine_similarity(user_embedding, symptom_embeddings)
        best_match_idx = np.argmax(similarities)

        response = {
            "disease": diseases[best_match_idx]["name"],
            "info": diseases[best_match_idx]["info"]
        }
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
