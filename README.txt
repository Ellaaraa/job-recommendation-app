DESCRIPTION:

This package contains this README.txt file, the 'DOC' folder, and the 'CODE' folder.

The 'DOC' folder contains 'team020report.pdf', which is the final report writeup, and 'team020poster.pdf', which is the final poster.

The 'CODE' folder contains 'app.py', which contains the main application, 'clean_data.py', which is the data cleaning script,'embeddings.py', which generates the word embeddings, 'requirements.txt', and other helper files.


INSTALLATION:

First, navigate to the 'CODE' directory
    > cd CODE


In the command line, create a virtual environment with conda or venv inside a temp folder, then activate it.
    > virtualenv venv


Activate the environment
    # On Windows
    > venv\Scripts\activate

    # On Linux/Mac
    > source venv/bin/activate


Then, install the requirements with pip
    > pip install -r requirements.txt

    
The dataset used in this project can be found here https://www.kaggle.com/datasets/ravindrasinghrana/job-description-dataset. Due to the lack of quality results in the original dataset, additional location data was used https://simplemaps.com/data/world-cities. Download and unzip both of these files and place them inside the current working directory.



EXECUTION:

Run the data cleaning script, which will produce 'data_sample.csv'. This will only need to be done once.
    > python clean_data.py


Generate the word embeddings, 'job_title_embeddings' and 'job_skills_embeddings.npy'. This will only need to be done once.
NOTE: this is very time consuming, use a GPU if possible. 
    > python embeddings.py 


Once the word embeddings are generated, run the app.
    > python app.py

This will run the app locally, which can be seen at http://127.0.0.1:8050

