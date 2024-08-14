"""This  module contains a class with methods to handle the data"""
import os
import re
import matplotlib.pyplot as plt
import pandas as pd

class DataHandler():
    """This class is used to handle the data

    It assumes that the data is in the data folder.

    It assumes that the files in the data folder are csv files

    It assumes that the csv files has only two columns. The first column 'text'
    and the second column should be named 'labels' regardless of how many
    labels there are.
    """
    def __init__(self):
        pass

    def get_all_files_in_directory(self, data_path) -> list[str]:
        """Goes through a directory and all sub directories and returns all the files in them

        Args:
            data_path (str): The path to the directory.

        This function a sub routine of concatenate_data
        """

        all_files = []
        for root, dirs, files in os.walk(data_path):
            for file in files:
                all_files.append(os.path.join(root, file))

        return all_files

    def concatenate_data(self, data_path: str) -> None:
        """Concatenates the data into one csv file"""

        all_files = self.get_all_files_in_directory(data_path)

        list_of_data_frames = ([pd.read_csv(file, index_col=None) for
                                file in all_files if file.endswith(".csv")])

        concatenated_data_frames = pd.concat(list_of_data_frames, ignore_index=True)

        # Save the DataFrame to a CSV file
        concatenated_data_frames.to_csv("concatenated_data/concatenated_data.csv", index=False)

    def see_df_head(self) -> None:
        """Prints the head of the data frame"""

        path_to_concat_data = "concatenated_data/concatenated_data.csv"
        data_frame = pd.read_csv(path_to_concat_data)

        print(data_frame.head())

    def see_data_size(self) -> None:
        """Prints the size of the data frame and the label counts

        This is to see if the data is balanced by seeing. The number of rows for each labels
        should not differ too much.
        """

        path_to_concat_data = "concatenated_data/concatenated_data.csv"
        data_frame = pd.read_csv(path_to_concat_data)

        length_of_data_frames = f"Number of rows in concatenated data frame: {len(data_frame)}"

        label_counts = data_frame.groupby('labels').size()
        label_counts = re.sub("dtype: int64", "", str(label_counts))
        label_counts = re.sub("labels", "Number of rows with the following labels "
                              "in the concatenated data:", str(label_counts))
        print(label_counts)
        print(length_of_data_frames)


    def histogram(self) -> None:
        """Plots a histogram of the labels

        The histogram is to see a grahpical distribution of the labels
        and by that we can see if the data is balanced by seeing if the
        bars are about the same hight.

        The size of  the bars should  not differ too much
        """

        path_to_concat_data = "concatenated_data/concatenated_data.csv"
        data_frame = pd.read_csv(path_to_concat_data)

        labels = data_frame["labels"]

        labels.hist(figsize=(10, 5))

        plt.show()  # install PyQt5 for this to work.


if __name__ == "__main__":
    DataHandler()
