from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
import pandas as pd
import torch

# GLOBAL VARIABLES TO SET
FILEPATH = "data_sample.csv"
#SAMPLE_SIZE = 300000

df = pd.read_csv(FILEPATH)
#tmp = df.sample(SAMPLE_SIZE)

def torch_to_np(embedding):
    """Helper to get around the "Numpy is not available" error Angela is getting"""
    arr =  np.array(embedding.tolist())
    assert embedding.shape == arr.shape
    return arr

print(f"Cuda available: {torch.cuda.is_available()}")

data_cleaned = df

# Combine relevant textual columns into one for each row
data_cleaned['Job Title / Role'] = data_cleaned[['Job Title', 'Role']].apply(
    lambda x: ' '.join(x.dropna()), axis=1
)

data_cleaned['Job Description / Skills'] = data_cleaned[['Job Description', 'skills', 'Responsibilities']].apply(
    lambda x: ' '.join(x.dropna()), axis=1)

# Load the model for generating sentence embeddings
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
if torch.cuda.is_available():
    model = model.to('cuda')

print("Generating embeddings")

# Generate embeddings for the job title/role and job description/skills
job_title_embeddings = model.encode(data_cleaned['Job Title / Role'].tolist(), convert_to_tensor=True).cpu().numpy()
job_desc_embeddings = model.encode(data_cleaned['Job Description / Skills'].tolist(), convert_to_tensor=True).cpu().numpy()
job_skills_embeddings = model.encode(data_cleaned['skills'].tolist(), convert_to_tensor=True).cpu().numpy()

print("Saving embeddings")
# Save embeddings as .npy files
np.save(f'job_title_embeddings.npy', job_title_embeddings)
# np.save(f'job_description_embeddings.npy', job_desc_embeddings)
np.save(f'job_skills_embeddings.npy', job_skills_embeddings)

# Save the dataframe (same row order as embeddings)
data_cleaned.to_csv(f'job_postings_metadata.csv')


