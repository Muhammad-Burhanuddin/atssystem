import os
import re
import cv2
import pytesseract
import fitz  # PyMuPDF
import pandas as pd
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords

# Download NLTK stopwords
nltk.download('stopwords')

# Initialize Flask App
app = Flask(__name__)
CORS(app)

# Set path to Tesseract OCR (if needed)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Function to extract text from images using Tesseract OCR
def extract_text_from_image(image_path):
    try:
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray)
        return text
    except Exception as e:
        print(f"Error extracting text from image: {e}")
        return ""

# Function to extract text from PDF files
def extract_text_from_pdf(pdf_path):
    try:
        pdf_document = fitz.open(pdf_path)
        text = ""
        for page in pdf_document:
            text += page.get_text()
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

# Function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
    return text

# Load and process resumes (PDFs and Images)
resume_dir = "resumes/"
resume_files = [f for f in os.listdir(resume_dir) if f.endswith(('.pdf', '.png', '.jpg', '.jpeg'))]

data = {'resume_path': [], 'resume_text': [], 'clean_resume': []}

for file in resume_files:
    file_path = os.path.join(resume_dir, file)

    # Extract text based on file type
    if file.endswith('.pdf'):
        extracted_text = extract_text_from_pdf(file_path)
    else:
        extracted_text = extract_text_from_image(file_path)

    # Clean the extracted text
    cleaned_text = clean_text(extracted_text)

    # Store data
    data['resume_path'].append(file_path)
    data['resume_text'].append(extracted_text)
    data['clean_resume'].append(cleaned_text)

df = pd.DataFrame(data)

# Save to CSV for future use
df.to_csv("resume_dataset.csv", index=False)

# Feature extraction with TF-IDF
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf.fit_transform(df['clean_resume'])

# Save the TF-IDF model for later use
with open("tfidf_vectorizer.pkl", "wb") as tfidf_file:
    pickle.dump(tfidf, tfidf_file)

# Function to find relevant resumes based on a given skill set
def find_relevant_resumes(skill_set, tfidf, tfidf_matrix, data):
    skill_set_clean = clean_text(skill_set)
    skill_set_vector = tfidf.transform([skill_set_clean])
    cosine_similarities = cosine_similarity(skill_set_vector, tfidf_matrix).flatten()
    relevant_indices = cosine_similarities.argsort()[-5:][::-1]  # Top 5 relevant resumes
    relevant_resumes = data.iloc[relevant_indices]
    relevant_scores = cosine_similarities[relevant_indices]
    return relevant_resumes, relevant_scores

# Flask API to process skills and return matching resumes
@app.route('/predict', methods=['POST'])
def predict():
    skill_set = request.json['skill_set']
    relevant_resumes, relevant_scores = find_relevant_resumes(skill_set, tfidf, tfidf_matrix, df)

    results = []
    for idx, score in zip(relevant_resumes.index, relevant_scores):
        results.append({
            'resume_path': df['resume_path'][idx],
            'score': score
        })

    return jsonify(results)

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
