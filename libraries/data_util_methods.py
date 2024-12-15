"""This  module contains a class with methods to handle the data"""

import os
import re
import sys
import matplotlib.pyplot as plt
import pandas as pd

class DataUtils:
    """This class is used to handle the data."""
    def __init__(self):
        self.__data_path = None
        self.__files_in_directory = None
        self.__concatenated_data = None
        self.__concatenated_data_path = None


    def __set_all_files_in_directory(self) -> None:
        """Helper method to find files.

        It goes through all the sub directories and returns all the files in them

        This method is used in prepare_data
        """

        all_files = []
        for root, dirs, files in os.walk(self.__data_path):
            for file in files:
                all_files.append(os.path.join(root, file))

        self.__files_in_directory = all_files


    def __concatenate_data(self) -> None:
        """Helper method to concatenate data into one csv file.

        This method is used in the prepare_data method"""

        list_of_data_frames = ([pd.read_csv(file, index_col=None) for
                                file in self.__files_in_directory if file.endswith(".csv")])

        concatenated_data_frames = pd.concat(list_of_data_frames, ignore_index=True)

        self.__concatenated_data = concatenated_data_frames


    def __save_concatenated_data(self) -> None:
        """Saves the concatenated data to a csv file

        This method requires that the prepare_data method has been used
        first.
        """

        if self.__concatenated_data is None:
            sys.exit("Use the prepare_data method first")

        self.__concatenated_data.to_csv(self.__concatenated_data_path, index=False)

    def prepare_data(self, data_path: str, concatenated_data_path: str) -> None:
        """Loads the data to the class.

        It sets the attributes of the class and saves the data to a csv file
        with the value of concatenated_data_path.

        Args:
            data_path (str): The path to the data
        """

        self.__data_path = data_path
        self.__set_all_files_in_directory()
        self.__concatenate_data()
        self.__concatenated_data_path = concatenated_data_path
        self.__save_concatenated_data()

    def get_csv_file_head(self) -> None:
        """Prints the head of the data frame."""

        if not os.path.exists(self.__concatenated_data_path):
            sys.exit("Use the prepare_data method first")

        data_frame = pd.read_csv(self.__concatenated_data_path)

        print(data_frame.head())

    def get_data_size_of_csv_file(self) -> None:
        """Prints the size of the data frame and the label counts

        This is to see if the data is balanced by seeing. The number of rows for each labels
        should not differ too much.
        """

        if not os.path.exists(self.__concatenated_data_path):
            sys.exit("Use the prepare_data method first")

        data_frame = pd.read_csv(self.__concatenated_data_path)

        length_of_data_frames = f"Number of rows in concatenated data frame: {len(data_frame)}"

        label_counts = data_frame.groupby('labels').size()
        label_counts = re.sub("dtype: int64", "", str(label_counts))
        label_counts = re.sub("labels", "Number of rows with the following labels "
                              "in the concatenated data:", str(label_counts))
        print(label_counts)
        print(length_of_data_frames)


    def histogram_of_csv_file(self) -> None:
        """Plots a histogram of the labels in the csv file.

        The histogram is to see a grahpical distribution of the labels
        and by that we can see if the data is balanced by seeing if the
        bars are about the same hight.

        The size of the bars should  not differ too much.

        If they do, use ComplementNB as it is better for dealing
        with imbalanced data.
        """

        if not os.path.exists(self.__concatenated_data_path):
            sys.exit("Use the prepare_data method first")

        data_frame = pd.read_csv(self.__concatenated_data_path)

        labels = data_frame["labels"]

        labels.hist(figsize=(10, 5))

        plt.show()  # install PyQt5 for this to work.
