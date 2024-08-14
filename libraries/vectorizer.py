"""This module contains a class for vecorizing the getting info on the sparse matrix"""

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

class Vectorizer:
    """Class for vectorizing the getting info on the sparse matrix

    It stores the vectorized objects in the vectorized_objects directory,
    where other methods can load them. This  means it does not need to be trained
    everytime one wants to see information about the sparse matrix
    """

    def __init__(self):
        pass
    @staticmethod
    def vectorizer(concatened_df_path: str) -> None:
        """Takes the concatenated data and vectorizes it

        Args:
            concatened_df_path (str): The path to the concatenated data frame

        It then dumps the vectorized objects in the vectorized_objects directory,
        where other methods can load them.
        """

        if concatened_df_path != str:
            ValueError("Argument must be a string")
        if len(concatened_df_path) != 1:
            ValueError("Please provide only one argument")
        if concatened_df_path == "concatenated_data/concatenated_data.csv":

            data_frame = pd.read_csv(concatened_df_path)

            text = data_frame["text"]
            labels = data_frame["labels"]

            # splits the dataframe into train and test

            input_train, input_test, y_train, y_test = train_test_split(
                text, labels, random_state=123)

            # # instanciate a count vectorizer object
            vectorizer = CountVectorizer()
            x_train = vectorizer.fit_transform(input_train)
            x_test = vectorizer.transform(input_test)

            joblib.dump(y_train, 'vectorized_objects/Ytrain.pkl')
            joblib.dump(y_test, 'vectorized_objects/Ytest.pkl')
            joblib.dump(x_train, 'vectorized_objects/Xtrain.pkl')
            joblib.dump(x_test, 'vectorized_objects/Xtest.pkl')
            joblib.dump(vectorizer, 'vectorized_objects/vectorizer.pkl')

    @staticmethod
    def get_info_on_sparse_matrix(argument: str) -> None:
        """Gets info on the sparse matrix

        Args:
            argument (str): The argument can be the following:
                shape: Prints the shape of the matrix (the number of rows and columns)
                sum_non_zeros: Prints the important values in the matrix are zeros
                percentage_non_zeros: Prints the percentage of non zeros in the matrix
        """

        x_train = joblib.load('vectorized_objects/Xtrain.pkl')


        if argument != str:
            ValueError("Argument must be a string")
        if len(argument) == 0:
            ValueError("Please provide only one argument")
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
