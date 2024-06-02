import os
import pandas as pd
import fitz  # PyMuPDF
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import pickle
from nltk.corpus import stopwords

# Download NLTK data files
nltk.download('stopwords')

# Directory containing PDF resumes
resume_dir = 'resumes/'

# Automatically get all PDF files in the directory
resume_files = [f for f in os.listdir(resume_dir) if f.endswith('.pdf')]

# Assuming labels are provided separately or default to 0
# You need to replace this with actual logic to get labels if available
labels = [0] * len(resume_files)  # Defaulting to 0 for demonstration

# Create a DataFrame
data = {'resume_path': [], 'label': []}

for file_name, label in zip(resume_files, labels):
    file_path = os.path.join(resume_dir, file_name)
    data['resume_path'].append(file_path)
    data['label'].append(label)

df = pd.DataFrame(data)

# Save to CSV
df.to_csv('resume_dataset.csv', index=False)
print('Dataset created successfully!')

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    try:
        pdf_document = fitz.open(pdf_path)
        text = ""
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
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

# Load data from the CSV file
data = pd.read_csv('resume_dataset.csv')
data['resume'] = data['resume_path'].apply(extract_text_from_pdf)
data['clean_resume'] = data['resume'].apply(clean_text)

# Feature extraction with TF-IDF
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf.fit_transform(data['clean_resume'])

# Function to find relevant resumes based on a given skill set
def find_relevant_resumes(skill_set, tfidf, tfidf_matrix, data):
    skill_set_clean = clean_text(skill_set)
    skill_set_vector = tfidf.transform([skill_set_clean])
    cosine_similarities = cosine_similarity(skill_set_vector, tfidf_matrix).flatten()
    relevant_indices = cosine_similarities.argsort()[-5:][::-1]  # Top 5 relevant resumes
    relevant_resumes = data.iloc[relevant_indices]
    relevant_scores = cosine_similarities[relevant_indices]
    return relevant_resumes, relevant_scores

# Function to interact with the user and get skill set input
def get_user_skill_set():
    skill_set = input("Enter the skill set: ")
    return skill_set

# Interactive script to filter resumes based on user-provided skill sets
if __name__ == "__main__":
    skill_set = get_user_skill_set()
    relevant_resumes, relevant_scores = find_relevant_resumes(skill_set, tfidf, tfidf_matrix, data)

    # Display relevant resumes and their scores
    print("\nTop Relevant Resumes:")
    for idx, score in zip(relevant_resumes.index, relevant_scores):
        print(f"Resume Path: {data['resume_path'][idx]}, Score: {score}")

    # Optional: Save the TF-IDF vectorizer and model (if needed for later use)
    with open('tfidf_vectorizer.pkl', 'wb') as tfidf_file:
        pickle.dump(tfidf, tfidf_file)
