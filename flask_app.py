from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
from nltk.corpus import stopwords

# Load the saved model and data
with open('tfidf_vectorizer.pkl', 'rb') as tfidf_file:
    tfidf = pickle.load(tfidf_file)

data = pd.read_pickle('resume_data.pkl')

# Function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
    return text

# Function to find relevant resumes based on a given skill set
def find_relevant_resumes(skill_set, tfidf, tfidf_matrix, data):
    skill_set_clean = clean_text(skill_set)
    skill_set_vector = tfidf.transform([skill_set_clean])
    cosine_similarities = cosine_similarity(skill_set_vector, tfidf_matrix).flatten()
    relevant_indices = cosine_similarities.argsort()[-5:][::-1]  # Top 5 relevant resumes
    relevant_resumes = data.iloc[relevant_indices]
    relevant_scores = cosine_similarities[relevant_indices]
    return relevant_resumes, relevant_scores

# Create Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS

@app.route('/predict', methods=['POST'])
def predict():
    skill_set = request.json['skill_set']
    relevant_resumes, relevant_scores = find_relevant_resumes(skill_set, tfidf, tfidf_matrix, data)
    results = []
    for idx, score in zip(relevant_resumes.index, relevant_scores):
        results.append({
            'resume_path': data['resume_path'][idx],
            'score': score
        })
    return jsonify(results)

if __name__ == "__main__":
    tfidf_matrix = tfidf.transform(data['clean_resume'])
    app.run(debug=True, host='0.0.0.0', port=5000)  # Ensure it runs on all IP addresses
