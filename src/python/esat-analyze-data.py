__author__ = "Stefan Klapf"

import datetime

"""
------------------------------------------------------------------------------------------------------------------------
esat-analyze-data.py
This application allows to analyze pcap files and predict the class.
For this purpose a previously compiled random forest classifier model is used.
Based on this, the features for the classification are extracted from the pcap file and passed to the random forest ensemble.
------------------------------------------------------------------------------------------------------------------------
"""
import argparse
import io
import os
import pickle
import subprocess
import sys
import warnings

import matplotlib
from tabulate import tabulate

warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas

matplotlib.use('Agg')  # Dependency for pyfiglet, must be called before importing pyfiglet.
from pyfiglet import Figlet
from utils.esatconfig import EsatConfig

config = EsatConfig()
ECHO_MAC = config.echo_mac
MODEL_DIR = config.model_dir
VOICE_CSV = config.VOICE_CSV
REPORTS_DIR = config.report_dir

parser = argparse.ArgumentParser(description='Echo Speech Analysis Toolkit - Analyze Data')
parser.add_argument('-file', metavar='file', type=str, default="empty",
                    help='the full filename of a single pcap file to analyze')
parser.add_argument('-dir', metavar='dir', type=str, default="empty",
                    help='path to a directory with pcap files to analyze')
args = parser.parse_args()


def render_welcome():
    """Prints the welcome screen"""
    f = Figlet(font='slant')
    print(f.renderText('ESAT - Analyze'))
    print("Echo Speech Analysis Toolkit - Analyze Data, Version 1.0")


def predict(file_path, model, feature):
    """Conducts a classification with an RandomForestClassifier
        :param file_path: The path to a single pcap file
        :param model: The model which is used for the classification
        :param feature: The features that will be extracted from the pcap. See esat-config for feature sets
        :return: None
    """
    try:
        pcap_features = get_features_from_pcap(file_path, feature)
        df = pandas.DataFrame(pcap_features)
        col_count = len(df.columns) - 1
        x = df.iloc[:, 0:col_count].values  # attributes

        mod_prediction = model.predict(x)

        return mod_prediction[0]

    except Exception as ex:
        print("Error while predicting data!")
        print(ex)


def get_features_from_pcap(pcap_path, features):
    """ Uses tcptrace to extract the features. Returns a pandas dataframe with the features
    https://linux.die.net/man/1/tcptrace
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
        df.drop(df.columns.difference(features), 1, inplace=True)

        data = pandas.DataFrame(df.agg(['sum'], axis=0, inplace=True))
        return data

    except Exception as ex:
        print("Failed to get features from pcap: " + str(pcap_path))
        print(ex)


def read_data_from_disk(path):
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
            return data
    except Exception as ex:
        print("Failed to load data from disk!")
        print(ex)


def analyze_single_file(file_path, table_output=False):
    """Analyses a single pcap file
     :param file_path: The full path to a file to analyze
     :param table_output: should print a table after each classification?
     :return: None """

    print("-" * 123)
    print("Starting classification...")
    print(" [-] File:     '{}'".format(file_path))
    print(" [-] Echo Mac: {}".format(ECHO_MAC))
    df = pandas.DataFrame(columns=['File',
                                   'ID All', 'ID A Only', 'ID B Only',
                                   'Speaker All', 'Speaker A Only', 'Speaker B Only',
                                   'Category All', 'Category A Only', 'Category B Only'])

    data = [file_path]

    model_id_all = read_data_from_disk(MODEL_DIR + "model_id_all.pkl")
    model_id_a = read_data_from_disk(MODEL_DIR + "model_id_a.pkl")
    model_id_b = read_data_from_disk(MODEL_DIR + "model_id_b.pkl")

    model_speaker_all = read_data_from_disk(MODEL_DIR + "model_speaker_all.pkl")
    model_speaker_a = read_data_from_disk(MODEL_DIR + "model_speaker_a.pkl")
    model_speaker_b = read_data_from_disk(MODEL_DIR + "model_speaker_b.pkl")

    model_cat_all = read_data_from_disk(MODEL_DIR + "model_cat_all.pkl")
    model_cat_a = read_data_from_disk(MODEL_DIR + "model_cat_a.pkl")
    model_cat_b = read_data_from_disk(MODEL_DIR + "model_cat_b.pkl")

    data.append(predict(file_path, model_id_all, config.FEATURES_ALL))
    data.append(predict(file_path, model_id_a, config.FEATURES_A_ONLY))
    data.append(predict(file_path, model_id_b, config.FEATURES_B_ONLY))

    data.append(predict(file_path, model_speaker_all, config.FEATURES_ALL))
    data.append(predict(file_path, model_speaker_a, config.FEATURES_A_ONLY))
    data.append(predict(file_path, model_speaker_b, config.FEATURES_B_ONLY))

    data.append(predict(file_path, model_cat_all, config.FEATURES_ALL))
    data.append(predict(file_path, model_cat_a, config.FEATURES_A_ONLY))
    data.append(predict(file_path, model_cat_b, config.FEATURES_B_ONLY))
    df.loc[len(df.index)] = data

    if table_output:
        print(tabulate(df, headers='keys', tablefmt='psql', showindex=False))
    print("Analysis of file is finished!")
    print("\n")
    return data


def check_args_and_start_predict():
    """checks the provided args and starts classification on the provided sources
     """
    if args.file == "empty" and args.dir == "empty":
        print("No input provided! Usage:")
        print("Start the program with -file path/to/single/pcap/file.pcap")
        print("Start the program with -dir path/to/directory/with/multiple/files/")

    if args.file != "empty":
        analyze_single_file(args.file, True)

    if args.dir != "empty":
        directory_path = args.dir
        if not directory_path.endswith("/"):
            directory_path = directory_path + "/"

        try:
            file_list = os.listdir(directory_path)
            df = pandas.DataFrame(columns=['File',
                                           'ID All', 'ID A Only', 'ID B Only',
                                           'Speaker All', 'Speaker A Only', 'Speaker B Only',
                                           'Category All', 'Category A Only', 'Category B Only'])

            for filename in file_list:
                df.loc[len(df.index)] = analyze_single_file(directory_path + filename)

            print(tabulate(df, headers='keys', tablefmt='psql', showindex=False))
            now = str(datetime.datetime.now()).replace(" ","-")
            df.to_csv('{}classification-result-{}.csv'.format(REPORTS_DIR,now))
            print("Report saved to: {}classification-result-{}.csv".format(REPORTS_DIR,now))
        except Exception as ex:
            print("Failed while prediction files in directory: {}".format(directory_path))
            print(ex)

    print("Analysis finished. Closing Program!")


def main():
    render_welcome()
    print("Started analysis on: " + str(datetime.datetime.now()))
    check_args_and_start_predict()
    print("Finished analysis on: " + str(datetime.datetime.now()))


if __name__ == '__main__':
    main()
