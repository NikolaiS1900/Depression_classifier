"""This module contains a class for vecorizing the getting info on the sparse matrix"""

import joblib
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm 

class Vectorizer:
    """Class for vectorizing the getting info on the sparse matrix

    It stores the vectorized objects in the vectorized_objects directory,
    where other methods can load them. This  means it does not need to be trained
    everytime one wants to see information about the sparse matrix
    """

    def __init__(self):
        self.data_frame = pd.read_csv("concatenated_data/concatenated_data.csv")

        self.text = self.data_frame["text"]
        self.labels = self.data_frame["labels"]

    def count_vectorize(self) -> None:
        """Takes the concatenated data and vectorizes it

        It then dumps the vectorized objects in the vectorized_objects directory,
        where other methods can load them.
        """

        # splits the dataframe into train and test
        input_train, input_test, y_train, y_test = train_test_split(
            self.text, self.labels, random_state=123)

        # # instanciate a count vectorizer object
        vectorizer = CountVectorizer()
        x_train = vectorizer.fit_transform(input_train)
        x_test = vectorizer.transform(input_test)

        joblib.dump(y_train, "vectorized_objects/Ytrain.pkl")
        joblib.dump(y_test, "vectorized_objects/Ytest.pkl")
        joblib.dump(x_train, "vectorized_objects/Xtrain.pkl")
        joblib.dump(x_test, "vectorized_objects/Xtest.pkl")
        joblib.dump(input_train, "vectorized_objects/input_train.pkl")
        joblib.dump(input_test, "vectorized_objects/input_test.pkl")
        joblib.dump(vectorizer, "vectorized_objects/vectorizer.pkl")

    def word2vec(self, model: str) -> None:
        list_of_sentences = []
        for sent in self.data_frame["text"]:
            list_of_sentences.append(sent.split())

        w2v_model = Word2Vec(list_of_sentences, vector_size=100, window=5, min_count=1, workers=4)
        w2v_words = list(w2v_model.wv.key_to_index)

        sent_vectors = []
        valid_indices = []
        for index, sent in enumerate(tqdm(list_of_sentences)):
            sent_vec = np.zeros(100)
            word_count = 0 # num of words with a valid vector in the sentence/tweet
            for word in sent: # for each word in a tweet/sentence
                if word in w2v_words:
                    vec = w2v_model.wv[word]
                    sent_vec += vec
                    word_count += 1
            if word_count != 0:
                sent_vec /= word_count
                sent_vectors.append(sent_vec)
                valid_indices.append(index)
            else:
                pass

        input_sentences = np.array(sent_vectors)
        # Transform features to be non-negative using MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        input_sentences_transformed = scaler.fit_transform(input_sentences)

        # Filter labels to match the valid sentences
        input_labels = self.data_frame["labels"].values[valid_indices]

        input_train, input_test, y_train, y_test = train_test_split(input_sentences_transformed, input_labels, test_size=0.30, stratify=input_labels)

        # splits the dataframe into train and test
        # input_train, input_test, y_train, y_test = train_test_split(
        #     input_sentences_transformed, input_labels, random_state=123)

        from sklearn.naive_bayes import ComplementNB
        from sklearn.naive_bayes import GaussianNB
        from sklearn.naive_bayes import BernoulliNB
        from sklearn.naive_bayes import MultinomialNB
        
        
        # Train a Complement Naive Bayes classifier
        nbmodel = ComplementNB()
        nbmodel.fit(input_train, y_train)

        # Evaluate the model (this does not belong here)
        accuracy = nbmodel.score(input_test, y_test)
        print(f"Accuracy: {accuracy}")

        # joblib.dump(y_train, "vectorized_objects/Ytrain.pkl")
        # joblib.dump(y_test, "vectorized_objects/Ytest.pkl")
        # joblib.dump(input_train, "vectorized_objects/input_train.pkl")
        # joblib.dump(input_test, "vectorized_objects/input_test.pkl")


    @staticmethod
    def get_info_on_sparse_matrix(argument: str) -> None:
        """Gets info on the sparse matrix

        Args:
            argument (str): The argument can be the following:
                shape: Prints the shape of the matrix (the number of rows and columns)
                sum_non_zeros: Prints the important values in the matrix are zeros
                percentage_non_zeros: Prints the percentage of non zeros in the matrix
        """

        x_train = joblib.load("vectorized_objects/Xtrain.pkl")

        if argument == "shape":
            # Normally, we want more rows than columns, but in this case it is not a problem

            # Get the number of rows and columns
            num_rows, num_cols = x_train.shape

            # Print the number of rows and columns
            shape_info = f"Number of rows: {num_rows}\nNumber of columns: {num_cols}"

            print(shape_info)

        # Most important values in the matrix are zeros
        if argument == "sum_non_zeros":
            sum_non_zeros = (x_train != 0).sum()

            print(sum_non_zeros)

        if argument == "percentage_non_zeros":
            percentage_non_zeros = (x_train != 0).sum() / np.prod(x_train.shape)

            print(percentage_non_zeros)
        else:
            ValueError("Argument must be one of the following: shape, sum_non_zeros, "
                        "percentage_non_zeros")

if __name__ == "__main__":
    Vectorizer
