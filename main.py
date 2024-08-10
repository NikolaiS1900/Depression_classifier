import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split


class classifier():
    def __init__(self):
        self.df = pd.read_csv("data/Depression/depressive_tweets_processed_cleaned.csv")
        self.inputs = self.df["text"]
        self.labels = self.df["labels"]

    def see_df_head(self):
        print(self.df.head())

    def see_data_size(self):
        print(len(self.df))

    def histogram(self):
        self.labels = self.df["labels"]

        # histogram to see the distribution of the labels
        # and by that we can see if the data is balanced by seeing if the bars are about the same hight.
        self.labels.hist(figsize=(10, 5))

        plt.show()  # install PyQt5

    def vectorizer(self) -> dict:

        dict_Xtrain_Xtest = {}
        # test train split

        input_train, input_test, Ytrain, Ytest = train_test_split(
            self.inputs, self.labels, random_state=123)
        

        # instanciate a count vectorizer object
        vectorizer = CountVectorizer()
        Xtrain = vectorizer.fit_transform(input_train)
        Xtest = vectorizer.transform(input_test)

        dict_Xtrain_Xtest["Xtrain"] = Xtrain
        dict_Xtrain_Xtest["Xtest"] = Xtest


        return dict_Xtrain_Xtest

    def get_info_on_sparse_matrix(self, argument: str) -> None:

        dict_Xtrain_Xtest = self.vectorizer()

        Xtrain = dict_Xtrain_Xtest["Xtrain"]

        if argument != str:
            ValueError("Argument must be a string")
        if len(argument) == 0:
            ValueError("Please provide only one argument")
        if argument == "shape":
            # Normally, we want more rows than columns, but in this case it is not a problem

            # Get the number of rows and columns
            num_rows, num_cols = Xtrain.shape

            # Print the number of rows and columns
            shape_info = f"Number of rows: {num_rows}\nNumber of columns: {num_cols}"

            print(shape_info)

        # Most values in the matrix are zeros
        if argument == "sum_non_zeros":
            sum_non_zeros = (Xtrain != 0).sum()

            print(sum_non_zeros)

        if argument == "percentage_non_zeros":
            percentage_non_zeros = (Xtrain != 0).sum() / np.prod(Xtrain.shape)

            print(percentage_non_zeros)



    def create_model(self):

        dict_Xtrain_Xtest = self.vectorizer()

        # Choose model
        model = MultinomialNB()

        return model

classifier = classifier()
classifier = classifier.get_info_on_sparse_matrix("percentage_non_zeros")


## Jeg skal lige hav Ytest og Ytrain på plads, se hvordan kursusholderen gør.