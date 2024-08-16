from libraries.data_handler import *
from libraries.vectorizer import *
from libraries.model_methods import *

# data_handler = DataHandler()
# data_handler.concatenate_data("data")
# data_handler.histogram()

# vec = Vectorizer()
# vec.vectorizer("concatenated_data/concatenated_data.csv")
# vec.get_info_on_sparse_matrix("percentage_non_zeros")

create_model = ModelMethods()
# create_model.create_model("BernoulliNB")
# create_model.create_model("CategoricalNB")
# create_model.create_model("MultinomialNB")
create_model.predict_new_string("MultinomialNB")
