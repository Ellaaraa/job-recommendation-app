# job-recommendation-app

This project is a job recommendation platform that enables users to apply up to 9 filters and assign weights to each. The algorithm identifies and returns jobs with the highest similarity scores based on the user’s selections.


The dataset used in this project is sourced from here: 
https://www.kaggle.com/datasets/ravindrasinghrana/job-description-dataset 

Due to the lack of quality results in the original dataset, additional location data was used https://simplemaps.com/data/world-cities. Download and unzip both of these files and place them inside the current working directory.


EXECUTION:

Run the data cleaning script, which will produce 'data_sample.csv'. This will only need to be done once.
    > python clean_data.py


Generate the word embeddings, 'job_title_embeddings' and 'job_skills_embeddings.npy'. This will only need to be done once.
NOTE: this is very time consuming, use a GPU if possible. 
    > python embeddings.py 


Once the word embeddings are generated, run the app.
    > python app.py

This will run the app locally, which can be seen at http://127.0.0.1:8050
