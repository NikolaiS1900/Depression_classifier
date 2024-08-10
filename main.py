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
        # and by that we can see if the data is balanced
        self.labels.hist(figsize=(10, 5))

        plt.show()  # install PyQt5

classifier = classifier()
classifier = classifier.histogram()