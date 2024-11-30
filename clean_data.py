# IMPORTS ----------------------------------------------------------------------
import numpy as np
import pandas as pd
import json

# DATA ----------------------------------------------------------------------

# Job posting data
# Source = https://www.kaggle.com/datasets/ravindrasinghrana/job-description-dataset
df = pd.read_csv("job_descriptions.csv") 

# World Cities Database by SimpleMaps.com
# Source = https://simplemaps.com/data/world-cities
city_df = pd.read_csv("worldcities.csv")
city_df = city_df.iloc[:,[0,4,2,3]]

# BENEFITS ----------------------------------------------------------------------
def clean_benefits(s):
    return s[1:-1].strip().replace("'", "")
df['Benefits'] = df['Benefits'].apply(clean_benefits)

# COMPANY PROFILE -------------------------------------------------------------
def process_company_profile(s):
    try:
        if pd.isnull(s):
            return {'Sector':None, 'Industry':None}
        else:
            return json.loads(s[0:s.find(',"City"')]+'}')
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in row: {s}")
        print(f"Error details: {e}")
        return None  # or return the original string, depending on how you want to handle this
subdf = df['Company Profile'].apply(process_company_profile).apply(pd.Series)
df = pd.concat([df.drop('Company Profile', axis = 1), subdf], axis=1)

# EXPERIENCE ----------------------------------------------------------------------

# Split the 'Experience' column into two new columns: 'Min Experience' and 'Max Experience'
df[['Min Experience', 'Max Experience']] = df['Experience'].str.extract(r'(\d+)\s+to\s+(\d+)')

# Convert the extracted strings into numeric values (integers)
df['Min Experience'] = pd.to_numeric(df['Min Experience'], errors='coerce')
df['Max Experience'] = pd.to_numeric(df['Max Experience'], errors='coerce')

# Handle rows that may have only one number (e.g., '10+ Years' might be represented as '0 to 10 Years')
# You can fill the missing values in 'Max Experience' with the same values from 'Min Experience' in such cases.
df['Max Experience'].fillna(df['Min Experience'], inplace=True)

# Insert the new columns right after the 'Experience' column
experience_index = df.columns.get_loc('Experience')  # Get the index of the 'Experience' column

# Insert 'Min Experience' and 'Max Experience' after the 'Experience' column
df.insert(experience_index + 1, 'Min Experience', df.pop('Min Experience'))
df.insert(experience_index + 2, 'Max Experience', df.pop('Max Experience'))

# QUALIFICATIONS --------------------------------------------------------------------------

# Convert all values in the 'Qualifications' column to uppercase for consistency
df['Qual Edit'] = df['Qualifications'].str.upper()

# Remove leading/trailing whitespace
df['Qual Edit'] = df['Qual Edit'].str.strip()

# Insert the new columns right after the 'Experience' column
qual_index = df.columns.get_loc('Qualifications')  # Get the index of the 'Experience' column

# Insert 'Min Experience' and 'Max Experience' after the 'Experience' column
df.insert(qual_index + 1, 'Qual Edit', df.pop('Qual Edit'))

# Define a mapping for qualifications to specific buckets
qualification_mapping = {
    'BA': 'Bachelor',
    'BBA': 'Bachelor',
    'BCA': 'Bachelor',
    'B.TECH': 'Bachelor',
    'B.COM': 'Bachelor',
    'MCA': 'Master',
    'M.COM': 'Master',
    'M.TECH': 'Master',
    'MBA': 'MBA',
    'PHD': 'PHD'
}

# Apply the mapping to the 'Qual Edit' column to create a new 'Qualification Bucket' column
df['Qual Edit'] = df['Qual Edit'].replace(qualification_mapping)

# SALARY --------------------------------------------------------------------------------------

df[['Min Salary in $', 'Max Salary in $']] = df['Salary Range'].str.extract(r'\$(\d+)K-\$(\d+)K')

# Convert the extracted values to integers and multiply by 1000 to get the full salary amount
df['Min Salary in $'] = df['Min Salary in $'].astype(int) * 1000
df['Max Salary in $'] = df['Max Salary in $'].astype(int) * 1000

# Insert 'Min Salary' and 'Max Salary' columns directly after 'Salary Range'
salary_index = df.columns.get_loc('Salary Range')
df.insert(salary_index + 1, 'Min Salary in $', df.pop('Min Salary in $'))
df.insert(salary_index + 2, 'Max Salary in $', df.pop('Max Salary in $'))

# COMPANY SIZE -------------------------------------------------------------------------

# Ensure data type consistency (convert to integer if necessary)
df['Company Size'] = df['Company Size'].astype(int)

# Create size buckets
check1 = df['Company Size'].max() - df['Company Size'].min()
partition1 = round(check1/3)
small = df['Company Size'].min() + partition1
medium = small + partition1
# print(partition1)
def categorize_company_size(size):
    if size < small:
        return 'Small'
    elif small <= size <= medium:
        return 'Medium'
    else:
        return 'Large'

# Apply the categorization function
df['Company Size Category'] = df['Company Size'].apply(categorize_company_size)

cosize_index = df.columns.get_loc('Company Size')
df.insert(cosize_index + 1, 'Company Size Category', df.pop('Company Size Category'))

# JOB POSTING DATE ---------------------------------------------------------------------
df['Job Posting Date'] = pd.to_datetime(df['Job Posting Date'], errors='coerce')

# Check if there are still invalid dates
invalid_dates = df[df['Job Posting Date'].isna()]

# DROP PREFERENCE ----------------------------------------------------------------------
df = df.drop('Preference', axis = 1)

# SECTOR -------------------------------------------------------------------------------
# Address three companies with missing sector information
df.loc[df['Company'] == 'Estée Lauder','Sector'] = "Consumer Goods"
df.loc[df['Company'] == 'Estée Lauder','Industry'] = "Cleaning products, perfumes, and toiletries"
df.loc[df['Company'] == "Dunkin'Brands Group, Inc.",'Sector'] = "Food and Beverage"
df.loc[df['Company'] == "Dunkin'Brands Group, Inc.",'Industry'] = "Food and beverage"
df.loc[df['Company'] == "Peter Kiewit Sons",'Sector'] = "Construction and engineering"
df.loc[df['Company'] == "Peter Kiewit Sons",'Industry'] = "Construction and engineering"

sector_mapping = {
    "Transportation/Logistics": ["transportation", "logistics", "airlines"],
    "Financial Services": ["financial", "banking", "investment", "insurance", "payroll"],
    "Healthcare/Pharmaceuticals": ["healthcare", "pharmaceuticals", "medical", "health"],
    "Travel/Hospitality": ["travel","hospitality"],
    "Media/Entertainment": ["media", "entertainment"],
    "Food/Beverage": ["food", "beverage", "restaurant"],
    "Manufacturing/Industrial": ["manufacturing", "industrial"],
    "Mining/Chemicals": ["mining", "chemical", "metal", "aluminum"],
    "Energy/Utilites": ["oil", "gas", "energy", "utilities", "water"],
    "Engineering" : ["engineering"],
    "Aerospace/Defense": ["aerospace", "defense", "security"],
    "Construction/Materials": ["construction", "infrastructure", "materials", "cement"],
    "Telecommunications": ["telecom", "communication"],
    "Appliances/Equipment": ["semiconductor", "appliance", "electronic", "equipment", "electrical"],
    "Technology": ["technology", "software", "it", "tech", "data"],
    "Retail": ["retail", "wholesale", "e-commerce", "consumer", "apparel"],
    "Automotive": ["automotive"]
}

def map_sector(text):
    if pd.isnull(text):
        return "Other"
    text = text.lower()  # Convert to lowercase for case-insensitive matching
    for sector, keywords in sector_mapping.items():
        if any(keyword in text for keyword in keywords):
            return sector
    return "Other"  # Default label if no keywords match

# Apply mapping to each row in the DataFrame
df["Sector Group"] = df["Sector"].apply(map_sector)
print(df.head(10))

# LOCATION  -----------------------------------------------------------------------------
# Original location columns contained many inaccuracies
# To correct this for the sake of testing the algorithm and deliverable, new locations were randomly chosen for each job

# Generate random locations
random_indices = np.random.randint(0, len(city_df), size=len(df))

# Extract the random rows from city_df
sampled_data = city_df.iloc[random_indices].to_numpy()

# Assign the sampled columns to the corresponding columns in df (location, Country, longitude, latitude)
df.iloc[:, 9:13] = sampled_data[:, :4]

# EXPORT --------------------------------------------------------------------------------
df_sample = df.sample(n=500000, random_state=42)  # random_state ensures reproducibility
df_sample.to_csv('data_sample.csv', encoding='utf-8', index = False)