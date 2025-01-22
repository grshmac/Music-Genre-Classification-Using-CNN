This is our project on ML where we choose Music Genre Classification as final year project

# Project: Music-Genre-Classification
-	We are working on Anaconda Prompt:

-	Go inside the project directory from anaconda

-   install all the requirements through this file
-	pip install -r requirements.txt

-   create a compatible environment 
-	conda create --name <your_env_name> python=3.10 -y

-   Now the environment is created and we are ready to go with our application
-   Interact through Jupyter-Notebook by opening notebook through anaconda prompt

# Get the dataset:
-	From Kaggle: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/data

#  To use Colab for the project ML implementation:
-   Sign in to Colab with your gmail account
-	Upload dataset folder, requirements.txt, and rest of files to your drive inside MGC folder.
-	Give access to drive
-   Run the .ipynb files

#  (To solely use the application) How to use the flask app:
-   Clone the git repo to VSCode application
-	Use python terminal, install flask
-   run the commands:
conda activate <your_env_name>
flask run
-	http://127.0.0.1:5000  go to this port
-	Then, it should show below page:

![alt text](applic_screenshots/image.png)

-	Give your audio input to the application through “Choose file” button
-	Then you easily get your prediction as:

![alt text](applic_screenshots/image-1.png)
