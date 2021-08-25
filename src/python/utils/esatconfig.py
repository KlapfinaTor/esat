__author__ = "Stefan Klapf"

import os

"""
------------------------------------------------------------------------------------------------------------------------
esatconfig.py
Helps to manage the config file and store constant shared values.
------------------------------------------------------------------------------------------------------------------------
"""

import configparser
import sys


class EsatConfig:
    def __init__(self):
        self.version = "1.0"
        self.config_filename = 'esat-config.ini'
        self.report_dir = './reports/'
        self.VOICE_CSV = "./voice_commands.csv"
        self.FEATURES_ALL = ['total_packets_a2b', 'total_packets_b2a',
                             'ack_pkts_sent_a2b', 'ack_pkts_sent_b2a',
                             'pure_acks_sent_a2b', 'pure_acks_sent_b2a',
                             'unique_bytes_sent_a2b', 'unique_bytes_sent_b2a',
                             'actual_data_pkts_a2b', 'actual_data_pkts_b2a',
                             'actual_data_bytes_a2b', 'actual_data_bytes_b2a',
                             'pushed_data_pkts_a2b', 'pushed_data_pkts_b2a',
                             'max_segm_size_a2b', 'max_segm_size_b2a',
                             'min_segm_size_a2b', 'min_segm_size_b2a',
                             'avg_segm_size_a2b', 'avg_segm_size_b2a',
                             'max_win_adv_a2b', 'max_win_adv_b2a',
                             'min_win_adv_a2b', 'min_win_adv_b2a',
                             'avg_win_adv_a2b', 'avg_win_adv_b2a',
                             'initial_window_bytes_a2b', 'initial_window_bytes_b2a',
                             'initial_window_pkts_a2b', 'initial_window_pkts_b2a',
                             'data_xmit_time_a2b', 'data_xmit_time_b2a',
                             'idletime_max_a2b', 'idletime_max_b2a',
                             'throughput_a2b', 'throughput_b2a']
        self.FEATURES_A_ONLY = ['total_packets_a2b',
                                'ack_pkts_sent_a2b',
                                'pure_acks_sent_a2b',
                                'unique_bytes_sent_a2b',
                                'actual_data_pkts_a2b',
                                'actual_data_bytes_a2b',
                                'pushed_data_pkts_a2b',
                                'max_segm_size_a2b',
                                'min_segm_size_a2b',
                                'avg_segm_size_a2b',
                                'max_win_adv_a2b',
                                'min_win_adv_a2b',
                                'avg_win_adv_a2b',
                                'initial_window_bytes_a2b',
                                'initial_window_pkts_a2b',
                                'data_xmit_time_a2b',
                                'idletime_max_a2b',
                                'throughput_a2b']

        self.FEATURES_B_ONLY = ['total_packets_b2a',
                                'ack_pkts_sent_b2a',
                                'pure_acks_sent_b2a',
                                'unique_bytes_sent_b2a',
                                'actual_data_pkts_b2a',
                                'actual_data_bytes_b2a',
                                'pushed_data_pkts_b2a',
                                'max_segm_size_b2a',
                                'min_segm_size_b2a',
                                'avg_segm_size_b2a',
                                'max_win_adv_b2a',
                                'min_win_adv_b2a',
                                'avg_win_adv_b2a',
                                'initial_window_bytes_b2a',
                                'initial_window_pkts_b2a',
                                'data_xmit_time_b2a',
                                'idletime_max_b2a',
                                'throughput_b2a']
        try:
            config = configparser.ConfigParser()
            config.read(self.config_filename)

            if "BASIC" not in config:
                print("Config File '{}' not found!".format(self.config_filename))
                print("Should be in the same dir as the application!")
                print("Closing Application!")
                sys.exit(1)

            self.echo_mac = config['BASIC']['EchoMACAddress']
            self.wlan_if_name = config['BASIC']['ETHInterface']
            self.capture_dir = config['DIRECTORIES']['CAPTURE_DIR']
            self.model_dir = config['DIRECTORIES']['MODEL_DIR']
            self.training_data_dir = config['DIRECTORIES']['TRAINING_DATA_DIR']
            self.aws_polly_key_id = config['AWS_POLLY']['AWS_ACCESS_KEY_ID']
            self.aws_polly_secret_key = config['AWS_POLLY']['AWS_SECRET_ACCESS_KEY']

            # Check dirs if they exist,if not create them.
            self.check_dir(self.capture_dir)
            self.check_dir(self.model_dir)
            self.check_dir(self.training_data_dir)
            self.check_dir(self.report_dir)

        except Exception as ex:
            print("Failed to load config file:" + self.config_filename)
            print(ex)
            print("Closing Application!")
            sys.exit(1)

    def check_dir(self, dir_to_check):
        """Checks if the directory exsits, if not create it
           :param dir_to_check: the directory that is checked
            :return: None
         """
        try:
            if not os.path.exists(dir_to_check):
                os.makedirs(dir_to_check)
        except Exception as ex:
            print("Error while checking or creating directory: '{}'".format(dir_to_check))
            print(ex)
