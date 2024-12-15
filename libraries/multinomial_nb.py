"""This module contains a class for creating models and showing info on models.

It can create models for BernoulliNB, ComplementNB and MultinomialNB
"""
import joblib
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.naive_bayes import MultinomialNB

class ModelMethods():
    "The class for creating models"
    def __init__(self):
        pass

    @staticmethod
    def create_model(model: str) -> None:
        """Creates models for BernoulliNB, ComplementNB and MultinomialNB

        Args:
            model (str): The model can be the following:
                BernoulliNB
                ComplementNB
                MultinomialNB
        """

        x_train = joblib.load("vectorized_objects/Xtrain.pkl")
        y_train = joblib.load("vectorized_objects/Ytrain.pkl")

        multinomial_nb_model = MultinomialNB()
        multinomial_nb_model_fitted = multinomial_nb_model.fit(x_train, y_train)

        joblib.dump(multinomial_nb_model_fitted, "created_models/MultinomialNB")