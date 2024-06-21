# Credit Card Defaulter Prediction
This project aims to predict credit card defaulters using machine learning models. The project is deployed on an AWS server and uses a variety of Python libraries for data manipulation, machine learning, and visualization.


## Project Overview
Credit card default prediction is crucial for financial institutions to minimize losses and manage credit risk effectively. This project involves the following steps:

 - Data Preprocessing
 - Exploratory Data Analysis (EDA)
 - Model Training and Evaluation
 - Model Deployment

## Project structure
    ├── .dvc
    ├── .github
    │   ├──workflow
    │       ├── main.yaml
    ├── artifacts
    │   ├── model.pkl
    │   ├── preprocessor.pkl
    │   ├── raw.csv
    │   ├── test.csv
    ├── venv
    ├── logs
    ├── mlruns
    ├── notebooks
    │   ├── data
    │        ├──uci_credit_card.csv
    │   ├── eda.ipynb
    │   ├── model_training.ipynb
    │   ├── research.ipynb
    ├── src
    │   ├── components
    │   │   ├── data_ingestion.py
    │   │   ├── data_transformation.py
    │   │   ├── model_evaluation.py
    │   │   ├── model_trainer.py
    │   ├── pipelines
    │   │   ├── prediction_pipeline.py
    │   │   ├── training_pipeline.py
    │   ├── utils
    │   │   ├── utils.py
    │   ├── logger.py
    │   ├── exception.py
    ├── templates
    │   ├── form.html
    │   ├── index.html
    │   ├── result.html
    ├── .gitignore
    ├── app.py
    ├── dockerfile
    ├── init_setup.sh
    ├── license
    ├── README.md
    ├── requirements.txt
    ├── setup.py
    ├── template.py
    └── test.py


## Getting Started

This will help you understand how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

## Installation Steps

### Installation from GitHub

Follow these steps to install and set up the project directly from the GitHub repository:

1. **Clone the Repository**
   - Open your terminal or command prompt.
   - Navigate to the directory where you want to install the project.
   - Run the following command to clone the GitHub repository:
     ```
     git clone https://github.com/rahulyadav7000/Credit-Card-Defaulter-Prediction.git
     ```

2. **Create a Virtual Environment** (Optional but recommended)
   - It's a good practice to create a virtual environment to manage project dependencies. Run the following command:
     ```
     conda create -p <Environment_Name> python==<python version> -y
     ```

3. **Activate the Virtual Environment** (Optional)
   - Activate the virtual environment based on your operating system:
       ```
       conda activate <Environment_Name>/
       ```

4. **Install Dependencies**
   - Navigate to the project directory:
     ```
     cd [project_directory]
     ```
   - Run the following command to install project dependencies:
     ```
     pip install -r requirements.txt
     ```

5. **Run the Project**
   - Start the project by running the appropriate command.
     ```
     python app.py
     ```

6. **Access the Project**
   - Open a web browser or the appropriate client to access the project.