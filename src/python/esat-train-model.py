__author__ = "Stefan Klapf"
"""
------------------------------------------------------------------------------------------------------------------------
esat-train-model.py
Trains the random forest models and generate some basic reports, confusion matrices, boxplots, etc...
------------------------------------------------------------------------------------------------------------------------
"""

import argparse
import datetime
import io
import os
import pickle
import shutil
import subprocess
import sys
import time
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
from pandas.core.common import SettingWithCopyWarning

warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)  # Hides the redundant pandas warning
warnings.simplefilter(action='ignore', category=FutureWarning)
# warnings.simplefilter(action='ignore', category=UserWarning)
import pandas

matplotlib.use('Agg')  # Dependency for pyfiglet, must be called before importing pyfiglet.
from pyfiglet import Figlet
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, cross_validate
from utils.esatconfig import EsatConfig

parser = argparse.ArgumentParser(description='Echo Speech Analysis Toolkit - Train Model')
args = parser.parse_args()

config = EsatConfig()
ECHO_DOT_MAC = config.echo_mac
TRAINING_DATA_DIR = config.training_data_dir
MODEL_DIR = config.model_dir
REPORT_DIR = config.report_dir


def render_welcome():
    f = Figlet(font='slant')
    print(f.renderText('ESAT - ML'))
    print("Echo Speech Analysis Toolkit - Training Model, Version {}".format(config.version))


def write_data_to_disk(data, path):
    """ Saves serialized data to disk.
        :param data: The Object which is saved to disk.
        :param path full path for the file to write
        :return: None
    """
    try:
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print("File successfully written to disk: '{}'".format(path))
    except Exception as ex:
        print("Failed to write file!")
        print(ex)


def read_data_from_disk(path):
    """ Reads serialized data from disk.
        :param path full path for the file to read
        :return: the read data
    """
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
            return data
    except Exception as ex:
        print("Failed to load data from disk!")
        print(ex)


def generate_model(data_frames, m_name):
    """Generates the random forest model and accompanying reports, matrices.
        :data_frames a pandas datafram that contains all preprocessed training data [feature1, feature2,...,class label]
        :m_name the name of the model, used for the filename of the reports etc.
        :return: none
    """
    print("Starting to generate model: {}".format(m_name))
    start = time.time()
    plot_name = m_name.replace('_', '-')
    df = pandas.concat(data_frames)
    col_count = len(df.columns) - 1
    x = df.iloc[:, 0:col_count - 1].values  # attributes
    y = df.iloc[:, col_count].values  # labels

    """Divide into training and test set.
     Test set is 20% of training data.
     Setting a fix random state ensures that the train-test splits are always deterministic!
     """
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)
    model = RandomForestClassifier(n_estimators=500, n_jobs=-1)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    # Feature importance
    df_rf = []
    for feature in zip(df.columns, model.feature_importances_):
        df_rf.append(feature)

    df_rf = pandas.DataFrame(df_rf, columns=['Variable', 'Feature_Importance'])
    df_rf = df_rf.sort_values('Feature_Importance', ascending=False)
    df_rf.to_csv(REPORT_DIR + m_name + "_feature_importance.csv", index=False)
    create_feature_importance_plot(df_rf, "{}-feature-importance".format(plot_name),
                                   plot_title="Feature importance for model: {}".format(m_name))

    # Confusion Matrix
    if 'speaker' in m_name:
        class_labels = ['Hans', 'Marlene', 'Vicki']
        conf_mat = confusion_matrix(y_test, y_pred, labels=class_labels)
        create_confusion_matrix_image(conf_mat, plot_name, class_labels)
    elif 'cat' in m_name:
        class_labels = ['fun', 'info', 'music', 'news', 'time', 'weather']
        conf_mat = confusion_matrix(y_test, y_pred, labels=class_labels)
        create_confusion_matrix_image(conf_mat, plot_name, class_labels)
    elif 'id' in m_name:
        class_labels = []
        i = 1
        while i <= 50:
            class_labels.append('{}'.format(i))
            i += 1
        conf_mat = confusion_matrix(y_test, y_pred, labels=class_labels)
        create_confusion_matrix_image(conf_mat, plot_name, class_labels)
    else:
        conf_mat = confusion_matrix(y_test, y_pred)

    # Cross Validate 10 fold
    cv = cross_validate(model, x, y, cv=10)
    cv_mean = cv['test_score'].mean()

    # Classification Report
    class_report = classification_report(y_test, y_pred)
    class_report = class_report + "\nCross Validation: {:.2f}".format(cv_mean) + "\nAccuracy: {}".format(
        accuracy_score(y_test, y_pred))
    write_data_to_disk(class_report, REPORT_DIR + m_name + "_classification_report.txt")

    # Print Summary
    print("Confusion Matrix:\n{}\n".format(conf_mat))
    print("Classification Report:\n {} \n".format(class_report))
    print("Cross Validation Mean: {}".format(cv_mean))

    end = time.time()
    print("Model generation done in: {:.2f} seconds.".format(end - start))
    return model


def create_feature_importance_plot(df, plot_name='feature-importance', fig_size_x=19, fig_size_y=15,
                                   plot_title='Visualization of feature importance per tcptrace feature'):
    """ Visualize a dataframe as a plot (Feature importance)
        Parameters
        ----------
        df : Pandas dataframe
        plot_name: name of the plot (filename)
        fig_size_x : int
        fig_size_y : int
        plot_title : str
    """
    plt.figure(figsize=(fig_size_x, fig_size_y))
    sn.barplot(x='Feature_Importance', y='Variable', data=df, color='#494949')  # horizontal
    plt.tick_params(labelsize=13)
    plt.ylabel('Tcptrace Feature')
    plt.xlabel('Feature_Importance', fontsize=20)
    plt.title(plot_title, fontsize=20)
    plt.savefig("{}{}.png".format(REPORT_DIR, plot_name))


def create_confusion_matrix_image(matrix, model_name, class_labels):
    """ Visualize a matrix and saves in to the file system
        Parameters
        ----------
        matrix : array with values
        model_name: name of model
        class_labels : str
    """
    if 'speaker' in model_name:
        matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]  # normalize the absolut values to 0.0 - 1.0
        plt.figure(figsize=(16, 7))
        sn.heatmap(matrix, annot=True, annot_kws={'size': 10}, cmap=plt.cm.Greens, linewidths=0.2)
    elif 'cat' in model_name:
        matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]  # normalize the absolut values to 0.0 - 1.0
        plt.figure(figsize=(16, 7))
        sn.heatmap(matrix, annot=True, annot_kws={'size': 10}, cmap=plt.cm.Greens, linewidths=0.2)
    elif 'id' in model_name:
        plt.figure(figsize=(20, 14))  # Visualize it as a heatmap
        sn.heatmap(matrix, annot=True, annot_kws={'size': 10}, cmap=plt.cm.Greens, linewidths=0.2)
    else:
        plt.figure(figsize=(20, 14))  # Visualize it as a heatmap
        sn.heatmap(matrix, annot=True, annot_kws={'size': 10}, cmap=plt.cm.Greens, linewidths=0.2)

    tick_marks = np.arange(len(class_labels))
    tick_marks2 = tick_marks + 0.5
    plt.xticks(tick_marks + 0.5, class_labels, rotation=25)
    plt.yticks(tick_marks2, class_labels, rotation=0)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix for Model: {}'.format(model_name), fontsize=20)
    plt.savefig("{}{}-confusion-matrix.png".format(REPORT_DIR, model_name))


def get_features_from_pcap(pcap_path, classification, features):
    """ Uses tcptrace to extract the features. Returns a pandas dataframe with the features
        https://linux.die.net/man/1/tcptrace
        :pcap_path Path to the pcap that gets analyzed
        :classification The class label
        :features A list of TCP features that are used for the training.
        :return Pandas dataframe with the features
    """
    try:
        tcp_comm = subprocess.Popen(
            "tcptrace -l --csv " + str(pcap_path) + " | tail -n +9", stdout=subprocess.PIPE, shell=True,
            universal_newlines=True)
        stdout = tcp_comm.communicate()

        if tcp_comm is None:
            print("Failed to start tcptrace. Closing application!")
            sys.exit(1)

        df_string = io.StringIO(stdout[0])
        df = pandas.read_csv(df_string, sep=",", skipinitialspace=True)
        df.drop(
            df.columns.difference(features), 1, inplace=True)

        data = pandas.DataFrame(df.agg(['sum'], axis=0, inplace=True))

        data.insert(len(features), "class", classification)
        return data

    except Exception as ex:
        print("Failed to get features from pcap: " + str(pcap_path))
        print(ex)


def process_pcaps(features, classification):
    """ Iterates over all folders in the trainings-data dir and extracts the wanted feature set.
        :features A list of TCP features that are used for the training.
        :classification The classification identifier, used to distinguish between models.
        :return A list with all extracted features with the corresponding class label.
    """
    all_dataframes = []
    print("-" * 123)
    print("[+] Started processing pcaps for classification " + str(classification))
    print("this can take a minute or two...")
    for directory in os.listdir(TRAINING_DATA_DIR):
        file_list = os.listdir(TRAINING_DATA_DIR + directory)
        for filename in file_list:
            if filename.endswith(".pcap"):
                class_id = ""
                if classification == "ID":
                    class_id = directory
                elif classification == "Speaker":
                    class_id = filename.split("_")[3]
                elif classification == "Category":
                    class_id = filename.split("_")[2]

                full_filename = TRAINING_DATA_DIR + directory + "/" + filename
                data = get_features_from_pcap(full_filename, class_id, features)
                all_dataframes.append(data)

    print("[+] Processing pcap files done!")
    print("-" * 123)
    return all_dataframes


def process_model(model_name, features, classification):
    """ Processes a single random forest model and saves the obtained model to the filesystem.
        :model_name name of the model that is being processed
        :features A list of TCP features that are used for the training.
        :classification The classification identifier, used to distinguish between models.
        :return none
    """
    try:
        model_path = MODEL_DIR + model_name + ".pkl"
        pcap_data_path = MODEL_DIR + "pcap_data_" + model_name + ".pkl"
        if os.path.exists(pcap_data_path):
            pcap_data = read_data_from_disk(pcap_data_path)
        else:
            pcap_data = process_pcaps(features, classification)
            write_data_to_disk(pcap_data, pcap_data_path)

        print("-" * 123)
        model = generate_model(pcap_data, model_name)
        write_data_to_disk(model, model_path)
    except Exception as ex:
        print("Failed to process model: {} with class: {}".format(model_name + ".pkl", classification))
        print(ex)


def process_all_models():
    """ Generate all 9 models needed for the classification
    :return none
    """
    process_model("model_id_all", config.FEATURES_ALL, "ID")
    process_model("model_id_a", config.FEATURES_A_ONLY, "ID")
    process_model("model_id_b", config.FEATURES_B_ONLY, "ID")

    process_model("model_speaker_all", config.FEATURES_ALL, "Speaker")
    process_model("model_speaker_a", config.FEATURES_A_ONLY, "Speaker")
    process_model("model_speaker_b", config.FEATURES_B_ONLY, "Speaker")

    process_model("model_cat_all", config.FEATURES_ALL, "Category")
    process_model("model_cat_a", config.FEATURES_A_ONLY, "Category")
    process_model("model_cat_b", config.FEATURES_B_ONLY, "Category")


def check_filesize_of_captures(move_outliers=False):
    """ Iterates over all captured samples and calculates the IQR distance per class label.
        Moves outlier to an extra folder. Generates numerous additional box plot visualisations for the samples.
        :return none
    """
    list_filesize = []
    for directory in os.listdir(TRAINING_DATA_DIR):
        file_list = os.listdir(TRAINING_DATA_DIR + directory)
        if directory == "0":
            continue
        for filename in file_list:
            if filename.endswith(".pcap"):
                full_filename = TRAINING_DATA_DIR + directory + "/" + filename
                capture_time = filename.split('_')[-1]
                capture_time = pandas.to_datetime(capture_time.split('.')[0])
                filesize = os.path.getsize(full_filename) / 1024  # in Kb
                list_filesize.append([directory, full_filename, filesize, capture_time])

    df = pandas.DataFrame(list_filesize, columns=['ID', 'Filename', 'Filesize', 'Datetime'])
    df['ID'] = df['ID'].astype(int)  # convert ID to int
    df['Filesize'] = df['Filesize'].astype(int)  # convert ID to int

    without_outliers = []
    i = 1
    while i <= 50:
        df_current_id = pandas.DataFrame(df.query('ID =={}'.format(i)),
                                         columns=['ID', 'Filename', 'Filesize', 'Datetime'])
        df_current_id.sort_values("Filesize", ascending=False, inplace=True)
        df_removed = remove_outliers_from_dataframe(df_current_id, 'Filesize')
        without_outliers.append(df_removed)
        i += 1

    df_no_outliers = pandas.concat(without_outliers)

    df.set_index('Filename', inplace=True)
    df_no_outliers.set_index('Filename', inplace=True)
    df_outliers_only = df[~df.apply(tuple, 1).isin(df_no_outliers.apply(tuple, 1))]
    df_outliers_only.to_csv('./outliers.csv')

    only_id_1 = df_no_outliers.query('ID ==1')
    only_id_1_with_outlier = df.query('ID ==1')

    plot_filesize(df_no_outliers, 'filesize-all-plot-no-outlier', 16, 7,
                  'Box plot visualization of all ID Class Labels (no outliers)')
    plot_filesize(df, 'filesize-all-plot-with-outlier', 16, 7,
                  'Box plot visualization of all ID Class Labels (with outliers)')
    plot_filesize(only_id_1, "filesize-id-1-plot-no-outlier", 7, 7,
                  'Box plot visualization of Class ID 1 (no outliers)')
    plot_filesize(only_id_1_with_outlier, "filesize-id-1-plot-with-outlier", 7, 7,
                  'Box plot visualization of Class ID 1 (with outliers)')

    mask_25_july_only = (df_outliers_only['Datetime'] > '2021-07-25 0:0:01') & (
            df_outliers_only['Datetime'] <= '2021-07-25 23:59:50')
    plot_histogram_24h(df_outliers_only.loc[mask_25_july_only], 'outliers-per-hour-on-25-july-plot',
                       title="Overview about the outliers per hour on the 25. July 2021")

    plot_histogram_captures_per_days(df, 'all-data-captures-by-day-in-july',
                                     title='Overview about the captures per day in July 2021')

    print("Size All Samples: {}".format(len(df.index)))
    print("Size Without Outliers: {}".format(len(df_no_outliers.index)))
    print("Size Outliers Only: {}".format(len(df_outliers_only.index)))
    if move_outliers:
        move_outlier_samples_to_folder(df_outliers_only, "./trainings_data_outliers/")


def move_outlier_samples_to_folder(df, target_dir):
    """ Moves the outliers to a seperate folder
        :return none
    """
    df = pandas.read_csv('./outliers.csv')  # Hack to get the correct columns, passing the df directly didnt work FIXME
    i = 1
    while i <= 50:  # FIXME create all 50 subfolders, find a cleaner way!
        os.makedirs(os.path.dirname("{}{}/".format(target_dir, i)), exist_ok=True)
        i += 1

    for file_path_old in df["Filename"]:
        shutil.move(file_path_old, file_path_old.replace("./trainings_data/", target_dir))


def plot_histogram_24h(df, plot_name, column_name='Datetime', color='#494949', title=''):
    """
        Visualize a dataframe as a histogram
        Parameters
        ----------
        df : Pandas dataframe
        plot_name: name of the plot (filename)
        column_name : str Column to visualize
        color : str
        title : str
    """
    plt.figure(figsize=(20, 10))
    ax = (df[column_name].groupby(df[column_name].dt.hour).count()).plot(kind="bar", color=color)
    ax.set_facecolor('#eeeeee')
    ax.set_xlabel("hour of the day")
    ax.set_ylabel("count")
    ax.set_title(title, fontsize=20)
    plt.savefig("{}{}.png".format(REPORT_DIR, plot_name))


def plot_histogram_captures_per_days(df, plot_name, column_name='Datetime', color='#494949', title=''):
    """
       Visualize a dataframe as a histogram
       Parameters
       ----------
       df : Pandas dataframe
       plot_name: name of the plot (filename)
       column_name : str Column to visualize
       color : str
       title : str
    """
    plt.figure(figsize=(20, 10))
    ax = (df[column_name].groupby(df[column_name].dt.day).count()).plot(kind="bar", color=color)
    ax.set_facecolor('#eeeeee')
    ax.set_xlabel("Day")
    ax.set_ylabel("Captured Samples")
    ax.set_title(title, fontsize=20)
    plt.savefig("{}{}.png".format(REPORT_DIR, plot_name))


def plot_filesize(df, plot_name='filesize_plot', fig_size_x=16, fig_size_y=7,
                  plot_title='Box plot visualization of sample filesize'):
    """
       Visualize a dataframe as a boxplot
       Parameters
       ----------
       df : Pandas dataframe
       plot_name: name of the plot (filename)
       fig_size_x : int
       fig_size_y : int
       plot_title : str
       """
    plt.figure(figsize=(fig_size_x, fig_size_y))
    sn.boxplot(x='ID', y='Filesize', data=df)
    plt.ylabel('Filesize in KB')
    plt.title(plot_title, fontsize=20)
    plt.savefig("{}{}.png".format(REPORT_DIR, plot_name))


def get_iqr_values_from_dataframe(df_in, col_name):
    """ Calculates the IQR distance for the provided df
        :return median, q1,q3,iqr,minimum,maximum
        """
    median = df_in[col_name].median()
    q1 = df_in[col_name].quantile(0.25)  # first quarter
    q3 = df_in[col_name].quantile(0.75)  # third quarter
    iqr = q3 - q1  # IQR
    minimum = q1 - 5.0 * iqr  # lower range, scale 5
    maximum = q3 + 5.0 * iqr  # upper range, scale 5
    return median, q1, q3, iqr, minimum, maximum


def remove_outliers_from_dataframe(df_in, col_name):
    """ Removes the outliers from the given dataframe
        :return dataframe without outliers
        """
    _, _, _, _, minimum, maximum = get_iqr_values_from_dataframe(df_in, col_name)
    df_out = df_in.loc[(df_in[col_name] > minimum) & (df_in[col_name] < maximum)]
    return df_out


def main():
    render_welcome()
    print("  [info] Training Data Dir '{}'".format(TRAINING_DATA_DIR))
    print("  [info] Model Data Dir '{}'".format(MODEL_DIR))

    try:
        print("Started training on: " + str(datetime.datetime.now()))
        # check_filesize_of_captures(False) #enable to check for outliers
        process_all_models()
        print("All models successfully trained and saved to disk.")
        print("Finished training on: " + str(datetime.datetime.now()))
    except Exception as ex:
        print("Failed to generate model!")
        print(ex)


if __name__ == '__main__':
    main()
