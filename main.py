from libraries.data_handler import DataHandler
from libraries.vectorizer import Vectorizer

data_handler = DataHandler()
data_handler.concatenate_data("data")
data_handler.histogram()

# vec = Vectorizer()
# vec.vectorizer("concatenated_data/concatenated_data.csv")
# vec.get_info_on_sparse_matrix("percentage_non_zeros")
