__author__ = "Stefan Klapf"

import datetime
import sys

"""
------------------------------------------------------------------------------------------------------------------------
esat-capture-data.py
This application helps to capture network traffic into pcap files. 
Uses the linux command line low level packet capture application dumpcap for it.
------------------------------------------------------------------------------------------------------------------------
"""

import argparse
import logging
import matplotlib

matplotlib.use('Agg')  # Dependency for pyfiglet, must be called before importing pyfiglet.
from signal import SIGINT
from subprocess import Popen
from pyfiglet import Figlet

from utils.esatconfig import EsatConfig

parser = argparse.ArgumentParser(description='Echo Speech Analysis Toolkit - Capture Data')
parser.add_argument('-filename', metavar='filename', type=str, default="empty",
                    help='the filename for the captured data')
parser.add_argument('-decrypt', metavar='decrypt', type=str, default="empty", help='decrypt the WPA2 packages')
args = parser.parse_args()

logging.getLogger("scapy.runtime").setLevel(logging.ERROR)

config = EsatConfig()

WLAN_IF_NAME = config.wlan_if_name
CAPTURE_DIR = config.capture_dir
ECHO_MAC = config.echo_mac
DEFAULT_CAPTURE_NAME = "esat-capture"


def prompt_get_filename():
    """Prompts the user to enter a filename for the capture. If no filename is provided the default one is used.
        :return: None
    """
    data = input("Enter a filename for the capture ({}):".format(DEFAULT_CAPTURE_NAME))
    if len(data) != 0:
        return data
    else:
        return DEFAULT_CAPTURE_NAME


def render_welcome():
    """Renders the welcome screen.
        :return: None
     """
    f = Figlet(font='slant')
    print(f.renderText('ESAT - Capture'))
    print("Echo Speech Analysis Toolkit - Capture, Version {}".format(config.version))


def check_capture_filename():
    """Logic for getting the filename. If no filename is provided via args, prompt the user
        :return: the filename for the capture
    """
    if args.filename == 'empty':
        tmp_filename = prompt_get_filename()
        if tmp_filename is not None:
            return tmp_filename
    else:
        return args.filename


def main():
    render_welcome()
    capture_name = check_capture_filename()

    try:
        # Dumpcap settings see: https://www.wireshark.org/docs/man-pages/dumpcap.html
        # Starts dumpcap process async
        proc_capture = Popen(
            ["dumpcap", "-i{}".format(WLAN_IF_NAME), "-w{}".format(CAPTURE_DIR + capture_name + ".pcap"), "-g", "-P",
             "-bfilesize:1000000", "-f", "ether host {}".format(ECHO_MAC)])  # Around 1000mb max per file capture

        if proc_capture is None:
            print("Failed to start Dumpcap. Closing application!")
            sys.exit(1)

        print("\n  [-] Echo Mac Address: {}".format(config.echo_mac))
        print("  [-] Capture started on: {}".format(datetime.datetime.now()))
        print("  [-] To stop the capture press 'CTRL+C'\n")
        proc_capture.wait()  # Wait till the dumpcap process finishes.
    except KeyboardInterrupt:
        print('Capture aborted!')
        proc_capture.send_signal(SIGINT)  # Sends term signal to the capture process


if __name__ == '__main__':
    main()
