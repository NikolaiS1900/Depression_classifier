import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
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

        dict_Xtrain_Xtest_Ytrain_Ytest = {}
        # test train split

        input_train, input_test, Ytrain, Ytest = train_test_split(
            self.inputs, self.labels, random_state=123)
        

        # instanciate a count vectorizer object
        vectorizer = CountVectorizer()
        Xtrain = vectorizer.fit_transform(input_train)
        Xtest = vectorizer.transform(input_test)

        dict_Xtrain_Xtest_Ytrain_Ytest["Xtrain"] = Xtrain
        dict_Xtrain_Xtest_Ytrain_Ytest["Xtest"] = Xtest
        dict_Xtrain_Xtest_Ytrain_Ytest["Ytrain"] = Ytrain
        dict_Xtrain_Xtest_Ytrain_Ytest["Ytest"] = Ytest
        dict_Xtrain_Xtest_Ytrain_Ytest["input_test"] = input_test
        dict_Xtrain_Xtest_Ytrain_Ytest["vectorizer"] = vectorizer



        return dict_Xtrain_Xtest_Ytrain_Ytest

    def get_info_on_sparse_matrix(self, argument: str) -> None:

        dict_Xtrain_Xtest_Ytrain_Ytest = self.vectorizer()

        Xtrain = dict_Xtrain_Xtest_Ytrain_Ytest["Xtrain"]

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



    def create_model(self) -> dict:

        dict_Xtrain_Xtest_Ytrain_Ytest = self.vectorizer()
        Xtrain = dict_Xtrain_Xtest_Ytrain_Ytest["Xtrain"]
        Ytrain = dict_Xtrain_Xtest_Ytrain_Ytest["Ytrain"]

        # Choose model
        model = MultinomialNB()
        model.fit(Xtrain, Ytrain)

        return model
    
    def show_model_info(self, scores: bool = True,
                        confusion_matrix: bool = True,
                        misclassified_classes: bool = True,
                        importantest_feauture: bool = True) -> None:

        #TODO: I need to be able to save the trained model, so I don't need to train it everytimne 
        # I want to do something with it.
        model = self.create_model()

        #TODO: I need to be able to save my vectorized data too·
        dict_Xtrain_Xtest_Ytrain_Ytest = self.vectorizer()
        Xtrain = dict_Xtrain_Xtest_Ytrain_Ytest["Xtrain"]
        Ytrain = dict_Xtrain_Xtest_Ytrain_Ytest["Ytrain"]
        Xtest = dict_Xtrain_Xtest_Ytrain_Ytest["Xtest"]
        Ytest = dict_Xtrain_Xtest_Ytrain_Ytest["Ytest"]
        inputs_test = dict_Xtrain_Xtest_Ytrain_Ytest["input_test"]
        vectorizer = dict_Xtrain_Xtest_Ytrain_Ytest["vectorizer"]


        Ptest = model.predict(Xtest)


        if confusion_matrix == True: # This helps us how good it performs on the different categories
            ConfusionMatrixDisplay.from_predictions(Ytest, Ptest)
            plt.show()

        if scores == True:
            print("train score:", model.score(Xtrain, Ytrain))
            print("test score:", model.score(Xtest, Ytest))

        if misclassified_classes == True:
        # Show some random misclassified examples
            np.random.seed(0)

            misclassified_idx = np.where(Ptest != Ytest)[0]
            if len(misclassified_idx) > 0:
                i = np.random.choice(misclassified_idx)
                print("True class:", Ytest.iloc[i])
                print("Predicted class:", Ptest[i])
                inputs_test.iloc[i]
                #print(misclassified_idx)
            else:
                print("No misclassified examples")

        if importantest_feauture == True: # Attempt to see which features "matters the most"
            #TODO: create stop word list and removes words like "the" and "a"
            class_features = model.feature_log_prob_.shape
            print(f"number of classes: {class_features[0]}")
            print(f"number of features: {class_features[1]}")

            # Show how the model have stored the classes
            order_of_classes = model.classes_
            print(order_of_classes)

            # The count vectorizer stores the mapping of each word and each feature index
            word_feature_index_map = vectorizer.vocabulary_
            #print(word_feature_index_map)

            # we actually want to have feature_index_word
            feature_index_word = vectorizer.get_feature_names_out()
            #print(feature_index_word)

            # assign the mapping in feature_index_word to idx2word
            idx2word = vectorizer.get_feature_names_out()
            #print(idx2word)

            # check the top 10 word in each class:
            #TODO: Make a forloop that goes through all the classes
            idx = np.argsort(-model.feature_log_prob_[0])[:10] # class 0
            top_ten_idx = idx2word[idx]
            print(top_ten_idx)

        else:
            pass
    def predict_new_string(self):

        model = self.create_model()

        # Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.
        Xnew = ["I feel so worthless", "I don't want to get up"]
        Xnew = np.reshape(Xnew, (-1, 1))



        prediction = model.predict(Xnew)
        print(prediction)


        # Predict probability for each class:
        probabilities = model.predict_proba(Xnew)
        probabilities


classifier = classifier()
#classifier = classifier.show_model_info(confusion_matrix=False, scores=False, misclassified_classes=True, importantest_feauture=True)
classifier = classifier.predict_new_string()


#Try with CategoriacalNB