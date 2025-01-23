import spacy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify
from flask_cors import CORS
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all domains

# Load English language model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

class CropDiseaseNLP:
    def __init__(self):
        # Disease database with symptoms and organic remedies
        self.disease_database = {
    "leaf_blight": {
        "symptoms": [
            "yellow leaves", "yellowing leaves", "brown spots",
            "wilting", "leaves turning yellow", "brown patches",
            "leaf spots", "dried leaves"
        ],
        "name": "Leaf Blight",
        "organic_remedies": [
            "Apply neem oil spray",
            "Use copper-based organic fungicide",
            "Remove infected leaves and burn them",
            "Improve air circulation between plants"
        ]
    },
    "powdery_mildew": {
        "symptoms": [
            "white powder", "white coating", "stunted growth",
            "leaf curling", "powdery coating", "white spots",
            "distorted leaves", "reduced growth"
        ],
        "name": "Powdery Mildew",
        "organic_remedies": [
            "Mix 1:10 milk to water solution and spray",
            "Apply baking soda solution (1 tbsp per gallon of water)",
            "Prune to improve air circulation",
            "Apply compost tea spray"
        ]
    },
    "root_rot": {
        "symptoms": [
            "wilting", "yellowing", "stunted growth", "soft roots",
            "rotting roots", "plant collapse", "brown roots",
            "mushy roots", "dying plant"
        ],
        "name": "Root Rot",
        "organic_remedies": [
            "Improve soil drainage",
            "Apply beneficial bacteria like Bacillus subtilis",
            "Add organic matter to soil",
            "Reduce watering frequency"
        ]
    },
    "rust": {
        "symptoms": [
            "orange spots", "red pustules", "yellow specks",
            "leaf discoloration", "powdery rust-like growth"
        ],
        "name": "Rust",
        "organic_remedies": [
            "Apply sulfur-based fungicide",
            "Prune infected leaves",
            "Improve ventilation",
            "Use neem oil"
        ]
    },
    "anthracnose": {
        "symptoms": [
            "dark lesions", "sunken spots", "brown patches",
            "fruit rot", "leaf blight"
        ],
        "name": "Anthracnose",
        "organic_remedies": [
            "Remove infected leaves",
            "Apply copper-based fungicide",
            "Maintain proper drainage",
            "Avoid overhead watering"
        ]
    },
    "fusarium_wilt": {
        "symptoms": [
            "wilting leaves", "yellowing stems", "vascular discoloration",
            "stunted growth", "plant collapse"
        ],
        "name": "Fusarium Wilt",
        "organic_remedies": [
            "Use disease-resistant seeds",
            "Sterilize soil before planting",
            "Apply biofungicides",
            "Improve drainage"
        ]
    },
    "downy_mildew": {
        "symptoms": [
            "yellow spots", "white fuzzy growth", "leaf curling",
            "stunted growth", "mildew on undersides"
        ],
        "name": "Downy Mildew",
        "organic_remedies": [
            "Apply copper spray",
            "Prune infected areas",
            "Avoid overhead irrigation",
            "Improve airflow"
        ]
    },
    "bacterial_blight": {
        "symptoms": [
            "water-soaked lesions", "leaf spots", "wilting",
            "blackened stems", "stem cankers"
        ],
        "name": "Bacterial Blight",
        "organic_remedies": [
            "Remove infected plants",
            "Spray copper-based fungicides",
            "Avoid overhead watering",
            "Plant resistant varieties"
        ]
    },
    "early_blight": {
        "symptoms": [
            "brown spots", "concentric rings", "leaf yellowing",
            "stem lesions", "fruit rot"
        ],
        "name": "Early Blight",
        "organic_remedies": [
            "Use compost tea spray",
            "Apply neem oil",
            "Mulch to prevent soil splash",
            "Remove infected leaves"
        ]
    },
    "late_blight": {
        "symptoms": [
            "dark lesions", "leaf curling", "white mildew",
            "fruit rot", "stem cankers"
        ],
        "name": "Late Blight",
        "organic_remedies": [
            "Spray copper fungicide",
            "Remove infected plants",
            "Improve drainage",
            "Avoid overcrowding plants"
        ]
    }
}

        
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(stop_words='english')
        
        # Create symptom corpus for vectorization
        self.symptom_corpus = []
        self.symptom_to_disease = {}
        for disease, info in self.disease_database.items():
            for symptom in info["symptoms"]:
                self.symptom_corpus.append(symptom)
                self.symptom_to_disease[symptom] = disease
        
        # Fit vectorizer with symptom corpus
        self.symptom_vectors = self.vectorizer.fit_transform(self.symptom_corpus)

    def preprocess_text(self, text):
        """Process input text to extract relevant information."""
        doc = nlp(text.lower())
        
        # Extract relevant phrases (nouns and adjectives)
        relevant_phrases = []
        for chunk in doc.noun_chunks:
            relevant_phrases.append(chunk.text)
        
        # Also add individual tokens that might be relevant
        relevant_tokens = [token.text for token in doc 
                         if token.pos_ in ['NOUN', 'ADJ'] 
                         and not token.is_stop]
        
        return relevant_phrases + relevant_tokens

    def find_matching_symptoms(self, input_text, threshold=0.3):
        """Find matching symptoms using NLP and cosine similarity."""
        # Preprocess input text
        processed_phrases = self.preprocess_text(input_text)
        
        # Convert to string for vectorization
        processed_text = " ".join(processed_phrases)
        
        # Vectorize input text
        input_vector = self.vectorizer.transform([processed_text])
        
        # Calculate similarity with all symptoms
        similarities = cosine_similarity(input_vector, self.symptom_vectors)
        
        # Find matching symptoms
        matching_symptoms = []
        for idx, similarity in enumerate(similarities[0]):
            if similarity > threshold:
                matching_symptoms.append(self.symptom_corpus[idx])
        
        return matching_symptoms

    def diagnose(self, input_text):
        """Diagnose disease based on input text description."""
        # Find matching symptoms
        matched_symptoms = self.find_matching_symptoms(input_text)
        
        if not matched_symptoms:
            return {
                "disease": "Unknown",
                "confidence": 0,
                "matched_symptoms": [],
                "remedies": ["Please provide more specific symptoms or consult an agricultural expert."]
            }
        
        # Count disease occurrences for matched symptoms
        disease_counts = {}
        for symptom in matched_symptoms:
            disease = self.symptom_to_disease[symptom]
            disease_counts[disease] = disease_counts.get(disease, 0) + 1
        
        # Find the disease with most matching symptoms
        best_match = max(disease_counts.items(), key=lambda x: x[1])
        disease_name = best_match[0]
        
        # Calculate confidence
        max_possible_symptoms = len(self.disease_database[disease_name]["symptoms"])
        confidence = (best_match[1] / max_possible_symptoms) * 100
        
        return {
            "disease": self.disease_database[disease_name]["name"],
            "confidence": round(confidence, 2),
            "matched_symptoms": matched_symptoms,
            "remedies": self.disease_database[disease_name]["organic_remedies"]
        }

# Initialize the diagnosis system
diagnosis_system = CropDiseaseNLP()

@app.route('/diagnose', methods=['POST'])
def diagnose_disease():
    """API endpoint for disease diagnosis"""
    if not request.json or 'symptoms' not in request.json:
        return jsonify({'error': 'No symptoms provided'}), 400
    
    symptoms = request.json['symptoms']
    result = diagnosis_system.diagnose(symptoms)
    return jsonify(result)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    # Get port from environment variable (Heroku sets this automatically)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
