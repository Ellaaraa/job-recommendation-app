import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from math import radians, cos, sin, sqrt, atan2, ceil
import warnings

# Suppress specific PyTorch UserWarnings
warnings.filterwarnings("ignore")

class Similarity:
    def __init__(self, jobs_df: pd.DataFrame):
        self.jobs_df = jobs_df
        
        # Define lookup matrices
        self.qualification_matrix = {
            "Bachelor": {"Bachelor": 1, "Master": 0.6, "MBA": 0.1, "PHD": 0.1},
            "Master": {"Bachelor": 0.8, "Master": 1, "MBA": 0.4, "PHD": 0.2},
            "MBA": {"Bachelor": 0.6, "Master": 0.8, "MBA": 1, "PHD": 0.4},
            "PHD": {"Bachelor": 0.4, "Master": 0.6, "MBA": 0.5, "PHD": 1}
        }

        work_type_scores = {
            'Intern': [1, 0.8, 0.6, 0.4, 0.3],
            'Temporary': [0.6, 1, 0.7, 0.8, 0.7],
            'Part-Time': [0.4, 0.6, 1, 0.5, 0.4],
            'Full-Time': [0.3, 0.5, 0.4, 1, 0.9],
            'Contract': [0.3, 0.7, 0.5, 0.8, 1]
        }
        self.work_type = pd.DataFrame(work_type_scores, index= ['Intern', 'Temporary', 'Part-Time', 'Full-Time', 'Contract'])

        company_size_scores = {
            'Small': [1, 0.4, 0.2],
            'Medium': [0.4, 1, 0.4],
            'Large': [0.2, 0.4, 1]
        }
        self.company_size = pd.DataFrame(company_size_scores, index=[
                                         'Small', 'Medium', 'Large'])

        # Load job embeddings
        self.job_title_embeddings = np.load('job_title_embeddings.npy', mmap_mode='r')
        self.job_skills_embeddings = np.load('job_skills_embeddings.npy', mmap_mode='r')

        # Initialize sentence transformer model
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


    def torch_to_np(self, embedding):
        arr = np.array(embedding.tolist())
        assert embedding.shape == arr.shape
        return arr
    

    def calculate_salary_score(self, row, user_salary):
        min_salary = row['Min Salary in $']
        max_salary = row['Max Salary in $']

        if user_salary <= min_salary:
            return 1
        if user_salary < max_salary:
            ratio = (max_salary - user_salary)/(max_salary - min_salary) 
            return ratio * (1 - 0.3) + 0.3
        else:
            passed_max = ceil((user_salary - max_salary)*0.1/10000)
            return 0.3 - min(0.3, passed_max)


    def calculate_experience_score(self, row, user_exp):
        min_experience = row['Min Experience']

        if user_exp >= min_experience:
            return 1
        else:
            penalty = -0.1 * (min_experience - user_exp)
            return max(0, 1 + penalty)


    def calculate_sector_score(self, listed_sector, choice1, choice2, choice3):
        if listed_sector == choice1:
            return 1
        if listed_sector == choice2:
            return 0.8
        if listed_sector == choice3:
            return 0.6
        return 0.2


    def haversine(self, lat1, lon1, lat2, lon2):
        # Haversine formula to calculate distance in km
        R = 6371  # Earth radius in kilometers
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        return R * c


    def location_similarity(self, row, user_lat, user_lon):
        # Location similarity function with distance-based score deduction
        job_lat = row['latitude']
        job_lon = row['longitude']

        # Calculate the distance between user and job location
        distance = self.haversine(user_lat, user_lon, job_lat, job_lon)

        # Start with full score of 1.0
        score = 1.0 - 0.005 * (distance // 25)

        # Ensure minimum score is zero
        return max(score, 0)


    def calculate_similarity_scores(self, user_example):
        """
        This function takes the Dataframe `df`, calculates the similarity score for each feature,
        and calculates the overall similarity score using the dot product between
        the user's desired weight and the similarity score for each feature

        This function returns none, as columns will be added to the df (in-place changes)

        parameters
        ---------
            user_example: (dict) a single user pulled from `users` dataframe

        returns
        ------
            results: (Pandas dataframe) a new dataframe that has the same columns as original
                    but with one new column per feature that the user specified
                    and a column for the final similarity score (aggregated)
        """
        df = self.jobs_df.copy(deep=True)
        feature_weights = []
        num_non_null = 0

        # SALARY
        if pd.notna(user_example['salary_min']):
            df['salary_score'] = df.apply(self.calculate_salary_score, axis=1, user_salary=user_example['salary_min'])
            feature_weights.append(user_example['salary_weight'])
            num_non_null += 1

        # EXPERIENCE
        if pd.notna(user_example['experience']):
            df['experience_score'] = df.apply(self.calculate_experience_score, axis=1, user_exp=user_example['experience'])
            feature_weights.append(user_example['experience_weight'])
            num_non_null += 1

        # QUALIFICATION
        if pd.notna(user_example['qualification']):
            qualification_lookup = pd.Series(self.qualification_matrix[user_example['qualification']])
            # Map the qualification matrix for each row (Vectorized lookup)
            df['qualification_score'] = df['Qual Edit'].map(qualification_lookup)
            feature_weights.append(user_example['qualification_weight'])
            num_non_null += 1

        # LOCATION
        if pd.notna(user_example['latitude']) and pd.notna(user_example['longitude']):
            df['location_score'] = df.apply(self.location_similarity, axis=1, user_lat=user_example['latitude'], user_lon=user_example['longitude'])
            feature_weights.append(user_example['location_weight'])
            num_non_null += 1

        # WORK TYPE
        if pd.notna(user_example['work_type']):
            df['work_type_score'] = df['Work Type'].map(self.work_type.loc[user_example['work_type']])
            feature_weights.append(user_example['work_type_weight'])
            num_non_null += 1

        # COMPANY SIZE
        if pd.notna(user_example['company_size']):
            df['company_size_score'] = df['Company Size Category'].map(self.company_size.loc[user_example['company_size']])
            feature_weights.append(user_example['company_size_weight'])
            num_non_null += 1

        # SECTOR
        if pd.notna(user_example['sector1']):
            df['sector_score'] = df['Sector Group'].apply(self.calculate_sector_score, args=(user_example['sector1'], user_example['sector2'], user_example['sector3']))
            if pd.notna(user_example['sector1']) or pd.notna(user_example['sector2']) or pd.notna(user_example['sector3']):
                feature_weights.append(user_example['sector_weight'])
                num_non_null += 1

        # JOB TITLE
        if pd.notna(user_example['job_title']):
            user_title_embedding = self.model.encode(user_example['job_title'], convert_to_tensor=True)
            # Compute cosine similarities for each embedding type
            title_similarities = cosine_similarity(self.torch_to_np(user_title_embedding).reshape(1, -1), self.job_title_embeddings)[0]
            feature_weights.append(user_example['job_title_weight'])
            df['title_score'] = title_similarities
            num_non_null += 1

        # JOB SKILLS
        if pd.notna(user_example['skills']):
            user_skills_embedding = self.model.encode(user_example['skills'], convert_to_tensor=True)
            skills_similarities = cosine_similarity(self.torch_to_np(user_skills_embedding).reshape(1, -1), self.job_skills_embeddings)[0]
            feature_weights.append(user_example['skills_weight'])
            df['skills_score'] = skills_similarities
            num_non_null += 1

        # Calculate overall final similarity score
        user_weights = np.array(feature_weights)
        df['final_similarity_score'] = np.dot(df.iloc[:,-num_non_null:], user_weights) / sum(user_weights)

        print(f"Number of features user specified: {num_non_null}")

        return df.sort_values('final_similarity_score', ascending=False).iloc[:100, :]


def main():
    df = pd.read_csv("data_sample.csv") # CHANGE THIS TO THE CORRECT FILE PATH
    similarity = Similarity(df)
    
    # Example User:
    user_example = dict(
        # salary_min = 100000, # numerical text entry only
        salary_min = np.nan,
        salary_weight = 3, #all weights should be 1-5 set at 3 default
        experience = 1, # numerical text entry only
        experience_weight = 3, #all weights should be 1-5 set at 3 default
        qualification = 'Bachelor', # dropdown (MBA, PHD, Bachelor, Master)
        qualification_weight = 4, #all weights should be 1-5 set at 3 default
        country = 'United States', # Not needed, if we stick to lat/lon
        city = 'Washington D.C.', # not needed, if we stick to lat/lon
        longitude = 38.9072, #these are floats
        latitude = 77.0369, #these are floats
        location_weight = 5, #all weights should be 1-5 set at 3 default
        work_type = "Intern", # dropdown (Intern, Part-Time, Full-Time, Contract)
        work_type_weight = 2, #all weights should be 1-5 set at 3 default
        company_size = 'Medium', # dropdown
        company_size_weight = 5,#all weights should be 1-5 set at 3 default
        job_title = 'Data Scientist', # text entry
        job_title_weight = 1,#all weights should be 1-5 set at 3 default
        skills = 'Python, SQL, scikit-learn, A/B Testing', # text entry
        skills_weight = 5,#all weights should be 1-5 set at 3 default
        sector1 = 'Healthcare/Pharmaceuticals', # Julia's select 3 options for choosing Top 3 Sectors
        # sector2 = 'Retail', # (it's okay if they don't select anything, or select less than 3)
        sector2 = np.nan,
        sector3 = np.nan,
        # sector3 = 'Financial Services',
        sector_weight = 2#all weights should be 1-5 set at 3 default
    )

    top100 = similarity.calculate_similarity_scores(user_example)
    print(top100.head())


if __name__ == "__main__":
    main()




