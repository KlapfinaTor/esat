__author__ = "Stefan Klapf"

import argparse
import datetime
import os
import subprocess
import sys
import time
from pathlib import Path

import boto3 as boto3
import pandas
from mutagen.mp3 import MP3

from utils.esatconfig import EsatConfig

config = EsatConfig()

VOICE_CSV = config.VOICE_CSV
DEFAULT_WAKE_WORD = "Alexa"
TRAINING_DATA_DIR = config.training_data_dir
IF_NAME = config.wlan_if_name
CAPTURE_DIR = config.capture_dir
ECHO_MAC = config.echo_mac

parser = argparse.ArgumentParser(description='Echo Speech Analysis Toolkit - Automatic Capture')
parser.add_argument('-iter', metavar='iter', type=int, default=1,
                    help='How many iterations will be conducted? Default :1')
args = parser.parse_args()


def generate_voice_commands_aws_polly(talker_voice_id, engine):
    print("[+] Starting to generate voices!")
    try:
        df = pandas.read_csv(VOICE_CSV, sep=";", skipinitialspace=True)
        for line in df.index:
            class_id = df['ID'][line]
            query_type = df['Type'][line]
            query_category = df['Category'][line]
            query = df['Query'][line]
            query_name_prepared = query.replace(" ", "_")
            config.check_dir("{}{}/".format(TRAINING_DATA_DIR, class_id))
            text_to_read = '<speak> {}, <break time="1s"/> {} <break time="2s"/> </speak>'.format(DEFAULT_WAKE_WORD,
                                                                                                  query)
            file_name = "{}{}/{}_{}_{}_{}_{}.mp3".format(TRAINING_DATA_DIR, class_id, class_id, query_type,
                                                         query_category, talker_voice_id,
                                                         query_name_prepared)
            generate_save_mp3_polly(file_name, talker_voice_id, engine, text_to_read)

        print("[+] Generate voices finished!")
    except Exception as ex:
        print("Failed to generate voices with polly!")
        print(ex)


def generate_save_mp3_polly(file_name, talker_voice_id, engine, text_to_read):
    try:
        if not os.path.exists(file_name):
            polly_client = boto3.client('polly',
                                        aws_access_key_id=config.aws_polly_key_id,
                                        aws_secret_access_key=config.aws_polly_secret_key,
                                        region_name='eu-west-2')
            response = polly_client.synthesize_speech(VoiceId=talker_voice_id,
                                                      Engine=engine,
                                                      OutputFormat='mp3',
                                                      Text=text_to_read,
                                                      TextType='ssml')
            file = open(file_name, 'wb')
            file.write(response['AudioStream'].read())
            file.close()
            print(file_name + " successfully generated!")
        else:
            print("'" + file_name + "' already exists, skipping...")
    except Exception as ex:
        print("Failed to generate voices with polly!")
        print(ex)


def generate_stop_aws_polly(talker_voice_id, engine):
    try:
        text_to_read = '<speak> {}, <break time="1s"/> stop <break time="2s"/> </speak>'.format(DEFAULT_WAKE_WORD)
        file_name = "{}{}/{}_Stop.mp3".format(TRAINING_DATA_DIR, "0", talker_voice_id)
        config.check_dir("{}{}/".format(TRAINING_DATA_DIR, "0"))
        generate_save_mp3_polly(file_name, talker_voice_id, engine, text_to_read)

        print("[+] Generate STOP voices finished!")
    except Exception as ex:
        print("Failed to generate STOP voices with polly!")
        print(ex)


def get_length_of_mp3(full_path):
    try:
        audio = MP3(full_path)
        audio_info = audio.info
        length_in_secs = int(audio_info.length)
        length_in_secs %= 3600
        return int(length_in_secs)
    except Exception as ex:
        print("Failed to get length of mp3!")
        print(ex)


def automatic_capture(talker_name):
    print("[+] Starting automatic capture!")
    try:
        for directory in os.listdir(TRAINING_DATA_DIR):
            if directory != "0":
                print("[*] Directory: " + directory)
                current_dir_path = TRAINING_DATA_DIR + directory + "/"
                file_list = os.listdir(current_dir_path)
                for filename in file_list:
                    if filename.endswith(".mp3") and filename.split("_")[3] == talker_name:
                        full_filename = TRAINING_DATA_DIR + directory + "/" + filename
                        filename_no_extension = Path(TRAINING_DATA_DIR + directory + "/" + filename).stem
                        duration = get_length_of_mp3(full_filename) + 15

                        print("   Starting capture for file: " + full_filename + " with duration: " + str(
                            duration) + " sec")

                        proc_capture = subprocess.Popen(
                            "timeout " + str(duration) + " dumpcap -i " + str(IF_NAME) + " -w " + str(
                                current_dir_path + filename_no_extension + ".pcap") + " -g -P -bfilesize:1000000 -f 'ether host {}'".format(
                                ECHO_MAC), shell=True)

                        print("\n  [-] Capture started on: {}".format(datetime.datetime.now()))
                        time.sleep(1)

                        if proc_capture is None:
                            print("Failed to start Dumpcap.")

                        print("Playing audio: " + full_filename)
                        play_audio = subprocess.Popen("ffplay -v warning -nodisp -autoexit {}".format(full_filename),
                                                      shell=True)
                        proc_capture.wait()  # Wait till the dumpcap process finishes.

                        if filename.split('_')[2] == "music":
                            # playing the vicki stop file, because the capture is halted before the "Alexa stop".
                            stop_file_name = "{}{}/{}_Stop.mp3".format(TRAINING_DATA_DIR, "0", "Vicki")
                            play_audio = subprocess.Popen(
                                "ffplay -v warning -nodisp -autoexit {}".format(stop_file_name),
                                shell=True)
                            time.sleep(5)

                        time.sleep(1)

        print("[+] Automatic capture finished!")
    except Exception as ex:
        print('Capture failed')
        print(ex)
    except KeyboardInterrupt as key:
        sys.exit(1)
        proc_capture.send_signal(SIGINT)


def main():
    generate_stop_aws_polly("Vicki", "neural")  # vicki only because the "alexa stop" command is not captured

    # Generate voices for each query in the voice_commands.csv
    generate_voice_commands_aws_polly("Vicki", "neural")
    generate_voice_commands_aws_polly("Marlene", "standard")
    generate_voice_commands_aws_polly("Hans", "standard")

    iter_counter = 0
    for x in range(args.iter):
        print("Doing iteration [{} / {}]".format(iter_counter, args.iter))
        # Start the automatic capture for the generated mp3 files.
        automatic_capture("Vicki")
        automatic_capture("Marlene")
        automatic_capture("Hans")
        print("Automatic capture finished!.")


if __name__ == '__main__':
    main()
