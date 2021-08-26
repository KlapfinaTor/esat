__author__ = "Stefan Klapf"

from sklearn.metrics import accuracy_score

"""
------------------------------------------------------------------------------------------------------------------------
esat-validate-helper.py
Allows additional reports and matrices to be generated from the trained models. Also conducts additional sample amount 
analysis.
------------------------------------------------------------------------------------------------------------------------
"""
import datetime
import os
import pickle
import time

import matplotlib
import pandas
import seaborn as sn
from matplotlib import pyplot as plt

matplotlib.use('Agg')  # Dependency for pyfiglet, must be called before importing pyfiglet.
from pyfiglet import Figlet
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_validate

from esatconfig import EsatConfig

config = EsatConfig()
MODEL_DIR = config.model_dir
REPORT_DIR = config.report_dir


def read_data_from_disk(path):
    """Reads serialized data from disk.
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


def write_data_to_disk(data, path):
    """Saves serialized data to disk.
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


def create_plot(data, plot_name='samples-report', fig_size_x=19, fig_size_y=15, plot_title='Title'):
    """ Visualize a dataframe as a plot (Feature importance)
       Parameters
       ----------
       data : Pandas dataframe
       plot_name: name of the plot (filename)
       fig_size_x : int
       fig_size_y : int
       plot_title : str
    """
    plt.figure(figsize=(fig_size_x, fig_size_y))

    if plot_name.split("-")[2] == "id":
        sn.lineplot(data=data.loc[data['model-name'] == 'pcap_data_model_id_all'], x="samples", y="accuracy",
                    palette="tab10",
                    linewidth=2.5, legend='brief')
        sn.lineplot(data=data.loc[data['model-name'] == 'pcap_data_model_id_a'], x="samples", y="accuracy",
                    palette="tab10",
                    linewidth=2.5, legend='brief')
        sn.lineplot(data=data.loc[data['model-name'] == 'pcap_data_model_id_b'], x="samples", y="accuracy",
                    palette="tab10",
                    linewidth=2.5, legend='brief')
        plt.legend(labels=['ID_All', 'ID_A', 'ID_B'], fontsize=20)

    elif plot_name.split("-")[2] == "speaker":
        sn.lineplot(data=data.loc[data['model-name'] == 'pcap_data_model_speaker_all'], x="samples", y="accuracy",
                    palette="tab10",
                    linewidth=2.5, legend='brief')
        sn.lineplot(data=data.loc[data['model-name'] == 'pcap_data_model_speaker_a'], x="samples", y="accuracy",
                    palette="tab10",
                    linewidth=2.5, legend='brief')
        sn.lineplot(data=data.loc[data['model-name'] == 'pcap_data_model_speaker_b'], x="samples", y="accuracy",
                    palette="tab10",
                    linewidth=2.5, legend='brief')
        plt.legend(labels=['Speaker_All', 'Speaker_A', 'Speaker_B'], fontsize=20)

    elif plot_name.split("-")[2] == "cat":
        sn.lineplot(data=data.loc[data['model-name'] == 'pcap_data_model_cat_all'], x="samples", y="accuracy",
                    palette="tab10",
                    linewidth=2.5, legend='brief')
        sn.lineplot(data=data.loc[data['model-name'] == 'pcap_data_model_cat_a'], x="samples", y="accuracy",
                    palette="tab10",
                    linewidth=2.5, legend='brief')

        sn.lineplot(data=data.loc[data['model-name'] == 'pcap_data_model_cat_b'], x="samples", y="accuracy",
                    palette="tab10",
                    linewidth=2.5, legend='brief')
        plt.legend(labels=['Cat_All', 'Cat_A', 'Cat_B'], fontsize=20)

    plt.tick_params(labelsize=13)
    plt.ylabel('Accuracy', fontsize=20)
    plt.xlabel('Samples', fontsize=20)
    plt.title(plot_title, fontsize=20)
    plt.savefig("{}{}.png".format(REPORT_DIR, plot_name))


def generate_model(data_frames, samples_per_class, model_name):
    """Generates the random forest model and accompanying reports, matrices.
        :data_frames a pandas dataframe that contains all preprocessed training data [feature1, feature2,...,class label]
        :m_name the name of the model, used for the filename of the reports etc.
        :return: none
    """

    start = time.time()
    df = pandas.concat(data_frames)
    df = df.groupby('class').head(samples_per_class).reset_index(drop=True)
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

    # Cross Validate 10 fold
    cv_mean = 0
    if samples_per_class <= 10:
        cv_mean = 0
    elif 10 < samples_per_class < 30:
        cv = cross_validate(model, x, y, cv=5, n_jobs=-1)
        cv_mean = cv['test_score'].mean()
    else:
        cv = cross_validate(model, x, y, cv=10, n_jobs=-1)
        cv_mean = cv['test_score'].mean()

    # Accuracy
    acc = accuracy_score(y_test, y_pred)

    # Print Summary
    print("Samples per class {}: {}".format(model_name, samples_per_class))
    print("Accuracy: {}".format(acc))
    print("Cross Validation Mean: {}".format(cv_mean))

    end = time.time()
    print("Model generation done in: {:.2f} seconds.".format(end - start))
    return model_name, samples_per_class, acc, cv_mean


def validate_all_models():
    """Trains the models with only a part of all available samples and generates reports.
            :return: None
         """
    models = list()
    for filename in os.listdir(MODEL_DIR):
        if filename.endswith(".pkl") and filename.split("_")[0] == "pcap":
            full_filename = MODEL_DIR + filename
            model_name = str(filename.split(".")[0])
            data = read_data_from_disk(full_filename)
            n_samples = [100]
            if model_name.split("_")[3] == "cat":
                n_samples = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 500, 1000, 2000, 2400]
            elif model_name.split("_")[3] == "id":
                n_samples = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 125, 150, 170, 200,
                             250, 300]
            elif model_name.split("_")[3] == "speaker":
                n_samples = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 500, 1000, 2000, 2400, 2800, 3000, 3500,
                             4000, 4700]

            for n in n_samples:
                models.append(generate_model(data, n, model_name))

    df = pandas.DataFrame(models, columns=['model-name', 'samples', 'accuracy', 'cross-validation-acc'])
    # Helper to skip the training part when all data was already calculated
    # write_data_to_disk(df, "./all-data-validate.pkl")
    # df = read_data_from_disk("./all-data-validate.pkl")
    df.to_csv("./validate-limited-samples-results.csv")
    create_plot(df, plot_name="samples-report-id",
                plot_title="Random forest accuracy based on limited samples: Model ID")
    create_plot(df, plot_name="samples-report-speaker",
                plot_title="Random forest accuracy based on limited samples: Model Speaker")
    create_plot(df, plot_name="samples-report-cat",
                plot_title="Random forest accuracy based on limited samples: Model Cat")


def render_welcome():
    f = Figlet(font='slant')
    print(f.renderText('ESAT - Validate'))
    print("Echo Speech Analysis Toolkit - Validate Helper, Version {}".format(config.version))


def main():
    render_welcome()
    try:
        print("Started validation on: " + str(datetime.datetime.now()))
        validate_all_models()
        print("Finished validation on: " + str(datetime.datetime.now()))
    except Exception as ex:
        print("Failed to validate models!")
        print(ex)


if __name__ == '__main__':
    main()
