"# Sentiment-analysis"

advised to use a virtual environment to run the project

Creating virtual environment using conda, run this command in the terminal in the directory in which your project is:
conda create -p venv python==3.10 -y

Activating the virtual environment, run the command in the same directory:
conda activate .\venv

Now install the dependencies in requirements.txt file with pip command:
pip install -r requirements.txt

Unzip the pickled algoriths with the same file name as pickled_algo in the same directory

Next run the web app:
streamlit run app.py
