import os
import sys
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
from urllib.parse import urlparse
from dataclasses import dataclass
import dagshub

from src.CreditCardDefaulter.logger import logging
from src.CreditCardDefaulter.exception import CustomException
from src.CreditCardDefaulter.utils.utils import load_object
import src.CreditCardDefaulter.utils.mlflow_setup as mlflow_setup


class ModelEvaluation:
    def __init__(self):
        pass

    def eval_metrics(self, model, features, label):
        accuracy = round(cross_val_score(model, features, label, cv=10, n_jobs=-1, scoring='accuracy').mean(), 2)
        f1 = round(cross_val_score(model, features, label, cv=10, n_jobs=-1, scoring='f1').mean(), 2)
        precision = round(cross_val_score(model, features, label, cv=10, n_jobs=-1, scoring='precision').mean(), 2)
        recall = round(cross_val_score(model, features, label, cv=10, n_jobs=-1, scoring='recall').mean(), 2)
        roc_auc = round(cross_val_score(model, features, label, cv=10, n_jobs=-1, scoring='roc_auc').mean(), 2)
        return accuracy, f1, precision, recall, roc_auc

    def initiate_model_evaluation(self, train_array, test_array):
        try:
            X_test, y_test = test_array[:, :-1], test_array[:, -1]
            model_path = os.path.join("artifacts", "model.pkl")
            model = load_object(model_path)

            mlflow.set_registry_uri("https://dagshub.com/rahulyadav7000/Credit-Card-Defaulter-Prediction.mlflow")
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme        
            print(tracking_url_type_store)
            
            dagshub.init(repo_owner='rahulyadav7000', repo_name='Credit-Card-Defaulter-Prediction', mlflow=True)


            with mlflow.start_run():
                mlflow.set_tag("Best Model", str(model).split("(")[0])
                
                accuracy, f1, precision, recall, roc_auc = self.eval_metrics(model, X_test, y_test)

                logging.info("accuracy_score: {}".format(accuracy))
                logging.info("f1_score: {}".format(f1))
                logging.info("precision_score: {}".format(precision))
                logging.info("recall_score: {}".format(recall))
                logging.info("roc_auc_score: {}".format(roc_auc))

                mlflow.log_metric("accuracy score", accuracy)
                mlflow.log_metric("f1 score", f1)
                mlflow.log_metric("precision score", precision)
                mlflow.log_metric("recall score", recall)
                mlflow.log_metric("roc_auc score", roc_auc)
                mlflow.sklearn.log_model(model, "model")
                
        except Exception as e:
            raise CustomException(e, sys)
