from libraries.data_handler import *
from libraries.vectorizer import *
from libraries.model_methods import *

# data_handler = DataHandler()
# data_handler.concatenate_data("data")
# data_handler.histogram()

vec = Vectorizer()
vec.vectorizer("concatenated_data/concatenated_data.csv")
# vec.get_info_on_sparse_matrix("percentage_non_zeros")

model_methods = ModelMethods()

model_methods.create_model("BernoulliNB")
model_methods.create_model("ComplementNB")
model_methods.create_model("MultinomialNB")

model_methods.show_model_info(
                        "ComplementNB",
                        scores = False,
                        show_confusion_matrix = False,
                        misclassified_classes = False,
                        importantest_feauture = False,
                        word_feature_index_map = False,
                        feature_index_word = False,
                        idx2word = False,
                        top10word = False
                        )

model_methods.predict_new_string("ComplementNB")
