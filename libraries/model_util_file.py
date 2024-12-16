import joblib

class ModelUtils():
    def __init__(self):
        self.__x_train_pkl_path = None
        self.__y_train_pkl_path = None
        self.__x_test_pkl_path = None
        self.__y_test_pkl_path = None
        self.__input_test_pkl_path = None
        self.__vectorizer_pkl_path = None
        self.__model_path = None


    def __check_pkl_paths(self, pkl_path: str) -> ValueError:
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
            self.__check_pkl_paths(x_train_pkl_path)
            self.__x_train_pkl_path = x_train_pkl_path
        if y_train_pkl_path:
            self.__check_pkl_paths(y_train_pkl_path)
            self.__y_train_pkl_path = y_train_pkl_path
        if x_test_pkl_path:
            self.__check_pkl_paths(x_test_pkl_path)
            self.__x_test_pkl_path = x_test_pkl_path
        if y_test_pkl_path:
            self.__check_pkl_paths(y_test_pkl_path)
            self.__y_test_pkl_path = y_test_pkl_path
        if input_test_pkl_path:
            self.__check_pkl_paths(input_test_pkl_path)
            self.__input_test_pkl_path = input_test_pkl_path
        if vectorizer_pkl_path:
            self.__check_pkl_paths(vectorizer_pkl_path)
            self.__vectorizer_pkl_path = vectorizer_pkl_path

    def get_pkl_paths(self) -> None:
        """Prints the pkl paths."""

        pkl_paths = (
            "The pkl paths are:\n"
            f"\t{self.__x_train_pkl_path}\n"
            f"\t{self.__y_train_pkl_path}\n"
            f"\t{self.__x_test_pkl_path}\n"
            f"\t{self.__y_test_pkl_path}\n"
            f"\t{self.__input_test_pkl_path}\n"
            f"\t{self.__vectorizer_pkl_path}"
        )

        print(pkl_paths)

    def set_model(self, model_path: str) -> None:
        """Set the model type"""
        self.__model_path = model_path

    def get_model(self) -> None:
        """Print the model"""
        print(self.__model_path)

    def ___load_model(self, model_path: str):
        """Helper method to load the the selected model

        Args:
            model_path (str): The path to the model

        Returns:
            The loaded model
        """

        loaded_model = joblib.load(model_path)

        return loaded_model
    
    def show_score(self):
        loaded_model = self.___load_model(self.__model_path)

        print("train score:", loaded_model.score(self.__x_train_pkl_path, self.__y_train_pkl_path))
        print("test score:", loaded_model.score(self.__x_test_pkl_path, self.__y_test_pkl_path))

