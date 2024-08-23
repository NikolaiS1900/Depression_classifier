"""This module contains a class for creating models and showing info on models.

It can create models for BernoulliNB, ComplementNB and MultinomialNB
"""
import joblib
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import ComplementNB
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


        if model == "BernoulliNB":

            bernoulli_nb_model = BernoulliNB()
            bernoulli_nb_model_fitted = bernoulli_nb_model.fit(x_train, y_train)

            # Save the model to a file
            joblib.dump(bernoulli_nb_model_fitted, "created_models/BernoulliNB")

        if model == "ComplementNB":

            complement_nb_model = ComplementNB()
            complement_nb_model_fitted = complement_nb_model.fit(x_train, y_train)

            joblib.dump(complement_nb_model_fitted, "created_models/ComplementNB")

        if model == "MultinomialNB":

            multinomial_nb_model = MultinomialNB()
            multinomial_nb_model_fitted = multinomial_nb_model.fit(x_train, y_train)

            joblib.dump(multinomial_nb_model_fitted, "created_models/MultinomialNB")


        else:
            ValueError("Argument is not a valid model. It must be one of the "
                       "following: BernoulliNB, ComplementNB, MultinomialNB")


    @staticmethod
    def show_model_info(model: str,
                        scores: bool = False,
                        show_confusion_matrix: bool = False,
                        misclassified_classes: bool = False,
                        importantest_feauture: bool = False,
                        word_feature_index_map: bool = False,
                        feature_index_word: bool = False,
                        idx2word: bool = False,
                        top10word: bool = False) -> None:
        """This helps us how good it performs on the different categories.

        It shows how many cases are classified correctly and how many are misclassified.

        Args:
            model (str): The model can be the following:
                BernoulliNB
                ComplementNB
                MultinomialNB
            scores (bool, optional): Shows the scores of the model.
            confusion_matrix (bool, optional): Shows the confusion matrix.
            misclassified_classes (bool, optional): Shows some random misclassified examples.
            importantest_feauture (bool, optional): Shows the most important features in the model.
            word_feature_index_map (bool, optional): Shows the word feature index map.
            feature_index_word (bool, optional): Shows the feature index word map.
            idx2word (bool, optional): Shows the index to word map.
            top10word (bool, optional): Shows the top 10 words in the model.

            Every argumenmt is False by default. It has to be set as True
            if you want it to be shown.
        """

        loaded_model = joblib.load(f"created_models/{model}")

        x_train = joblib.load("vectorized_objects/Xtrain.pkl")
        y_train = joblib.load("vectorized_objects/Ytrain.pkl")
        x_test = joblib.load("vectorized_objects/Xtest.pkl")
        y_test = joblib.load("vectorized_objects/Ytest.pkl")
        input_test = joblib.load("vectorized_objects/input_test.pkl")
        vectorizer = joblib.load("vectorized_objects/vectorizer.pkl")

        p_test = loaded_model.predict(x_test)

        if scores is True:
        # Show scores of the model
            print("train score:", loaded_model.score(x_train, y_train))
            print("test score:", loaded_model.score(x_test, y_test))

        if show_confusion_matrix is True:
        #  shows how many cases are classified correctly and how many are misclassified.
            ConfusionMatrixDisplay.from_predictions(y_test, p_test)
            plt.show()

        if misclassified_classes is True:
        # Show some random misclassified examples
            np.random.seed(0)

            misclassified_idx = np.where(p_test != y_test)[0]
            if len(misclassified_idx) > 0:
                i = np.random.choice(misclassified_idx)
                print("True class:", y_test.iloc[i])
                print("Predicted class:", p_test[i])
                print(f"\nMisclassified input:\n\n{input_test.iloc[i]}\n")
                print("\nIdx of misclassified input:\n\t", misclassified_idx)
            else:
                print("No misclassified examples")

        if importantest_feauture is True: # Attempt to see which features "matters the most"
            # TODO: create stop word list and removes words like "the" and "a"
            class_features = loaded_model.feature_log_prob_.shape
            print(f"number of classes: {class_features[0]}")
            print(f"number of features: {class_features[1]}")

            # Show how the model have stored the classes
            order_of_classes = loaded_model.classes_
            print(order_of_classes)

        if word_feature_index_map is True:

            # The count vectorizer stores the mapping of each word and each feature index
            word_feature_index_map = vectorizer.vocabulary_
            print(word_feature_index_map)

        if feature_index_word is True:

            # we actually want to have feature_index_word
            feature_index_word = vectorizer.get_feature_names_out()
            print(feature_index_word)

        if idx2word is True:

            # assign the mapping in feature_index_word to idx2word
            idx2word = vectorizer.get_feature_names_out()
            print(idx2word)

        if top10word is True:

            idx = np.argsort(-loaded_model.feature_log_prob_[0])[:10]
            idx2word = vectorizer.get_feature_names_out()
            top_ten_idx = idx2word[idx]
            print(top_ten_idx)

        else:
            pass

    @staticmethod
    def predict_new_string(model: str) -> None:
        """Test the model on a a list of string(s).

        It will take the new string and make a prediction
        on what class it belongs to.
        
        Args:
            model (str): The model can be the following:
                BernoulliNB
                ComplementNB
                MultinomialNB

            Select a model to make the prediction with.
        """

        user_input = input("Write something here and press enter: ")

        new_input = [user_input]

        loaded_model = joblib.load(f"created_models/{model}")
        vectorizer = joblib.load("vectorized_objects/vectorizer.pkl")

        # Transform the new inputs using the same vectorizer
        x_new = vectorizer.transform(new_input)

    # Predict the class labels for the new inputs
        p_new = loaded_model.predict(x_new)

        # Print the predicted class labels
        print("Predicted classes:", p_new)

        # Predict probability for each class:
        probabilities = loaded_model.predict_proba(x_new)

        print(probabilities)

if __name__ == "__main__":
    ModelMethods
