"""This module contains a class for creating models and showing info on models.

It can create models for BernoulliNB, CategoricalNB and MultinomialNB
"""
import joblib
import numpy as np
import scipy.sparse
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import MultinomialNB

class ModelMethods():
    "The class for creating models"
    def __init__(self):
        pass

    @staticmethod
    def create_model(model: str) -> None:
        """Creates models for BernoulliNB, CategoricalNB and MultinomialNB

        Args:
            model (str): The model can be the following:
                BernoulliNB
                CategoricalNB
                MultinomialNB
        """

        x_train = joblib.load("vectorized_objects/Xtrain.pkl")
        y_train = joblib.load("vectorized_objects/Ytrain.pkl")


        if model == "BernoulliNB":

            bernoulli_nb_model = BernoulliNB()
            bernoulli_nb_model_fitted = bernoulli_nb_model.fit(x_train, y_train)

            # Save the model to a file
            joblib.dump(bernoulli_nb_model_fitted, "created_models/BernoulliNB")

        if model == "CategoricalNB":

            # Check if training data is a sparse matrix
            if scipy.sparse.issparse(x_train) is True:

                 # Convert sparse data to dense data
                x_train = x_train.toarray()
                categorical_nb_model = CategoricalNB()
                categorical_nb_model_fitted = categorical_nb_model.fit(x_train, y_train)

                joblib.dump(categorical_nb_model_fitted, "created_models/CategoricalNB")

            # Check if training data is a numpy array
            if isinstance(x_train, np.ndarray) is True:

                categorical_nb_model = CategoricalNB()
                categorical_nb_model_fitted = categorical_nb_model.fit(x_train, y_train)

                joblib.dump(categorical_nb_model_fitted, "created_models/CategoricalNB")

            else:
                ValueError("Training data is not a sparse matrix or a numpy array")

        if model == "MultinomialNB":

            multinomial_nb_model = MultinomialNB()
            multinomial_nb_model_fitted = multinomial_nb_model.fit(x_train, y_train)

            joblib.dump(multinomial_nb_model_fitted, "created_models/MultinomialNB")


        else:
            ValueError("Argument is not a valid model. It must be one of the "
                       "following: BernoulliNB, CategoricalNB, MultinomialNB")
            
    
    # @staticmethod
    # def show_model_info(model: str,
    #                     scores: bool = True,
    #                     confusion_matrix: bool = True,
    #                     misclassified_classes: bool = True,
    #                     importantest_feauture: bool = True,
    #                     word_feature_index_map: bool = True,
    #                     feature_index_word: bool = True,
    #                     idx2word: bool = True,
    #                     top10word: bool = True) -> None:


    #     model = joblib.load(f"created_models/{model}")

    #     #TODO: I need to be able to save my vectorized data too·
    #     x_train = joblib.load("vectorized_objects/Xtrain.pkl")
    #     y_train = joblib.load("vectorized_objects/Ytrain.pkl")
    #     x_test = joblib.load("vectorized_objects/Xtest.pkl")
    #     y_test = joblib.load("vectorized_objects/Ytest.pkl")
    #     inputs_test = joblib.load("input_test.pkl")
    #     vectorizer = joblib.load("vectorized_objects/vectorizer.pkl")

    #     p_test = model.predict(x_test)

    #     if confusion_matrix == True: # This helps us how good it performs on the different categories
    #         ConfusionMatrixDisplay.from_predictions(y_test, p_test)
    #         plt.show()

    #     if scores == True:
    #         print("train score:", model.score(x_train, y_train))
    #         print("test score:", model.score(x_test, y_test))

    #     if misclassified_classes == True:
    #     # Show some random misclassified examples
    #         np.random.seed(0)

    #         misclassified_idx = np.where(p_test != y_test)[0]
    #         if len(misclassified_idx) > 0:
    #             i = np.random.choice(misclassified_idx)
    #             print("True class:", y_test.iloc[i])
    #             print("Predicted class:", p_test[i])
    #             inputs_test.iloc[i]
    #             #print(misclassified_idx)
    #         else:
    #             print("No misclassified examples")

    #     if importantest_feauture == True: # Attempt to see which features "matters the most"
    #         #TODO: create stop word list and removes words like "the" and "a"
    #         class_features = model.feature_log_prob_.shape
    #         print(f"number of classes: {class_features[0]}")
    #         print(f"number of features: {class_features[1]}")

    #         # Show how the model have stored the classes
    #         order_of_classes = model.classes_
    #         print(order_of_classes)

    #     if word_feature_index_map == True:

    #         # The count vectorizer stores the mapping of each word and each feature index
    #         word_feature_index_map = vectorizer.vocabulary_
    #         print(word_feature_index_map)

    #     if feature_index_word == True:

    #         # we actually want to have feature_index_word
    #         feature_index_word = vectorizer.get_feature_names_out()
    #         print(feature_index_word)

    #     if idx2word == True:

    #         # assign the mapping in feature_index_word to idx2word
    #         idx2word = vectorizer.get_feature_names_out()
    #         print(idx2word)

    #     if top10word == True:

    #         # check the top 10 word in each class:
    #         #TODO: Make a forloop that goes through all the classes
    #         idx = np.argsort(-model.feature_log_prob_[0])[:10] # class 0
    #         top_ten_idx = idx2word[idx]
    #         print(top_ten_idx)

    #     else:
    #         pass

    @staticmethod
    def predict_new_string(model: str):
    
        loaded_model = joblib.load(f"created_models/{model}")
        vectorizer = joblib.load("vectorized_objects/vectorizer.pkl")

        new_inputs = ["I feel so fucking sad again", "I am so gratefull for my programming skills", "I feel like shit"]


        # Transform the new inputs using the same vectorizer
        x_new = vectorizer.transform(new_inputs)

        if model == "CategoricalNB":

            # Check if training data is a sparse matrix
            if scipy.sparse.issparse(x_new) is True:

                 # Convert sparse data to dense data
                x_new = x_new.toarray()
                p_new = loaded_model.predict(x_new)

                print("Predicted classes:", p_new)

                # Predict probability for each class:
                probabilities = loaded_model.predict_proba(x_new)
                print(probabilities)

            # Check if training data is a numpy array
            elif isinstance(x_new, np.ndarray) is True:

                p_new = loaded_model.predict(x_new)
                print("Predicted classes:", p_new)

                # Predict probability for each class:
                probabilities = loaded_model.predict_proba(x_new)
                print(probabilities)

            else:
                pass

        if model != "CategoricalNB":

        # Predict the class labels for the new inputs
            p_new = loaded_model.predict(x_new)

            # Print the predicted class labels
            print("Predicted classes:", p_new)

            # Predict probability for each class:
            probabilities = loaded_model.predict_proba(x_new)

            print(probabilities)

        else:
            TypeError("Model must be CategoricalNB, BernoulliNB or MultinomialNB")

if __name__ == "__main__":
    ModelMethods
