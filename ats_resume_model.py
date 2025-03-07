import os
import pandas as pd
import fitz  # PyMuPDF
import re
import pytesseract
import cv2
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import pickle
from nltk.corpus import stopwords

# Download NLTK stopwords
nltk.download('stopwords')

# Paths to data directories
resume_dir = 'resumes/'
image_dir = 'resume_images/'

# Get all PDF and image files
resume_files = [f for f in os.listdir(resume_dir) if f.endswith('.pdf')]
image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    try:
        pdf_document = fitz.open(pdf_path)
        text = ""
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text += page.get_text("text")  # Extract text from page
        return text
    except Exception as e:
        print(f"Error extracting text from PDF {pdf_path}: {e}")
        return ""

# Function to extract text from an image using Tesseract OCR
def extract_text_from_image(image_path):
    try:
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        text = pytesseract.image_to_string(gray)  # Extract text using OCR
        return text
    except Exception as e:
        print(f"Error extracting text from image {image_path}: {e}")
        return ""

# Function to clean extracted text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
    return text

# Create a DataFrame to store resumes from both PDFs and images
data = {'resume_path': [], 'resume_text': [], 'source': []}

# Process PDFs
for file_name in resume_files:
    file_path = os.path.join(resume_dir, file_name)
    text = extract_text_from_pdf(file_path)
    data['resume_path'].append(file_path)
    data['resume_text'].append(text)
    data['source'].append('pdf')

# Process images
for file_name in image_files:
    file_path = os.path.join(image_dir, file_name)
    text = extract_text_from_image(file_path)
    data['resume_path'].append(file_path)
    data['resume_text'].append(text)
    data['source'].append('image')

# Convert data to a Pandas DataFrame
df = pd.DataFrame(data)
df['clean_resume'] = df['resume_text'].apply(clean_text)

# Save DataFrame as JSON
df.to_json('resume_data.json', orient='records', indent=4)
print("Resume data saved as JSON!")

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf.fit_transform(df['clean_resume'])

# Save TF-IDF model for later use
with open('tfidf_vectorizer.pkl', 'wb') as tfidf_file:
    pickle.dump(tfidf, tfidf_file)

# Function to find relevant resumes based on skill set
def find_relevant_resumes(skill_set, tfidf, tfidf_matrix, df):
    skill_set_clean = clean_text(skill_set)
    skill_set_vector = tfidf.transform([skill_set_clean])
    cosine_similarities = cosine_similarity(skill_set_vector, tfidf_matrix).flatten()
    relevant_indices = cosine_similarities.argsort()[-5:][::-1]  # Top 5 relevant resumes
    relevant_resumes = df.iloc[relevant_indices]
    relevant_scores = cosine_similarities[relevant_indices]
    return relevant_resumes, relevant_scores

# Interactive script to filter resumes based on user input
if __name__ == "__main__":
    skill_set = input("Enter the skill set: ")
    relevant_resumes, relevant_scores = find_relevant_resumes(skill_set, tfidf, tfidf_matrix, df)

    # Display relevant resumes in JSON format
    results = []
    for idx, score in zip(relevant_resumes.index, relevant_scores):
        if score > 0.0:
            results.append({
                "resume_path": df['resume_path'][idx],
                "score": round(score, 4),
                "source": df['source'][idx]
            })
    
    print(json.dumps(results, indent=4))  # Pretty print results
