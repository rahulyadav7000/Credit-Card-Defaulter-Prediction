import os
import sys
from src.CreditCardDefaulter.components.data_transformation import DataTransformation
from src.CreditCardDefaulter.components.model_trainer import ModelTrainer
from src.CreditCardDefaulter.logger import logging
from src.CreditCardDefaulter.exception import CustomException
import pandas as pd

from src.CreditCardDefaulter.components.data_ingestion import DataIngestion


obj = DataIngestion()

train_data_path, test_data_path = obj.initiate_data_ingestion()
data_transformation = DataTransformation()
train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)

model_trainer = ModelTrainer()
best_model, best_model_score = model_trainer.initiate_model_training(train_arr, test_arr)


model_eval_obj = ModelEvaluation()
model_eval_obj.initiate_model_evaluation(train_arr,test_arr)
