

class utils():
    def __init__(self):
        self.x_train_pkl = None
        self.y_train_pkl = None
        self.x_test_pkl = None
        self.y_test_pkl = None
        self.input_test_pkl = None
        self.vectorizer_pkl = None


    def check_pkl_paths(self, pkl_path: str) -> ValueError:
        """Helper method to check if paths end with pkl.

        This method is used in the set_pkl_paths method.

        Args:
            pkl_path (str): The path to the pkl file

        Returns:
            ValueError: if the path does not end with .pkl
        """

        if not pkl_path.endswith(".pkl"):
            raise ValueError("path must end with .pkl")

    def set_pkl_paths(self,
                      x_train_pkl_path: str,
                      y_train_pkl_path: str,
                      x_test_pkl_path: str,
                      y_test_pkl_path: str,
                      input_test_pkl_path: str,
                      vectorizer_pkl_path: str) -> None:

        """Setter method for the pkl paths

        Args:
            x_train_pkl_path (str): The path to the x_train_pkl file
            y_train_pkl_path (str): The path to the y_train_pkl file
            x_test_pkl_path (str): The path to the x_test_pkl file
            y_test_pkl_path (str): The path to the y_test_pkl file
            input_test_pkl_path (str): The path to the input_test_pkl file
            vectorizer_pkl_path (str): The path to the vectorizer_pkl file
        """

        if x_train_pkl_path:
            self.check_pkl_paths(x_train_pkl_path)
            self.x_train_pkl = x_train_pkl_path
        if y_train_pkl_path:
            self.check_pkl_paths(y_train_pkl_path)
            self.y_train_pkl = y_train_pkl_path
        if x_test_pkl_path:
            self.check_pkl_paths(x_test_pkl_path)
            self.x_test_pkl = x_test_pkl_path
        if y_test_pkl_path:
            self.check_pkl_paths(y_test_pkl_path)
            self.y_test_pkl = y_test_pkl_path
        if input_test_pkl_path:
            self.check_pkl_paths(input_test_pkl_path)
            self.input_test_pkl = input_test_pkl_path
        if vectorizer_pkl_path:
            self.check_pkl_paths(vectorizer_pkl_path)
            self.vectorizer_pkl = vectorizer_pkl_path

    def show_score(self):
        print("train_pkl score:", loaded_model.score(x_train_pkl, y_train_pkl))
        print("test_pkl score:", loaded_model.score(x_test_pkl, y_test_pkl))
