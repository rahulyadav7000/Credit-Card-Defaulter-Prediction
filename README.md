# Credit Card Defaulter Prediction
This project aims to predict credit card defaulters using machine learning models. The project is deployed on an AWS server and uses a variety of Python libraries for data manipulation, machine learning, and visualization.


## Project Overview
Credit card default prediction is crucial for financial institutions to minimize losses and manage credit risk effectively. This project involves the following steps:

Data Preprocessing
Exploratory Data Analysis (EDA)
Model Training and Evaluation
Model Deployment

# Project structure
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
