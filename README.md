# Echo Speech Analysis Toolkit (ESAT)

- Author: Stefan Klapf

## Idea Overview

The Echo Speech Analysis toolkit was developed as part of my master's thesis. It allows a fully automated interaction
with an Amazon Alexa System and interception of the resulting network traffic. In addition, the ESAT takes care of the
training and classification of the gathered network data. From the captured network traffic, TCP features are extracted,
which are used to train 9 different random forest models. The primary aim of the solution was to automate the capture
and training process for the gathered network data.

## Toolkit Feature Overview:

The toolkit is divided into several python applications:

- [esat-capture-data](./src/python/esat-capture-data.py): Allows to capture and save network packages as pcap files.
- [esat-automatic-capture](./src/python/esat-automatic-capture.py): Allows a fully automated capture of alexa
  interactions.
- [esat-train-model](./src/python/esat-train-model.py): Trains 9 different ML model based on the provided training data.
- [esat-analyze-data](./src/python/esat-analyze-data.py): Analyzes pcap files classifys them using the previously
  trained random forest model.
- [esat-validate-helper](./src/python/utils/esat-validate-helper.py): Allows additional reports and matrices to be
  generated from the trained models.

## Requirements

Requirements for installing und using the ESAT:

- Linux Operating System: The tool was developed and tested on Ubuntu 20.04
- Amazon Echo Dot Gen3 or Gen4

## SETUP

The tool was tested on Ubuntu 20.04. It requires a user with sudoer privileges to be run because it uses linux command
line applications to capture network traffic from an interface.

Install the following linux package dependencies:

```bash
# sudo apt install python3              // Python 3
# sudo apt install python3-pip          // Pip3 packet manager
# sudo apt install wireshark-common     // Dumcap Tool
# sudo apt install tcptrace             // Tcptrace
# sudo apt install ffmpeg               // For playing audio
```

Run pip3 in the "./src/python/" directory to install the required python dependencies:

```bash
# sudo pip3 install -r requirements.txt
```

Edit the ``esat-config.ini``:

Fill the "EchoMACAddress" and "WLANInterfaceName" with your values. Add the AWS keys if you intend to use the automatic
generation of sample audio files.

```ini
[BASIC]
EchoMACAddress = ff:ff:ff:ff:ff:ff
ETHInterface = enp6s0
[AWS_POLLY]
AWS_ACCESS_KEY_ID = YOUR_AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY = YOUR_SECRET_AWS_ACCESS_KEY
[DIRECTORIES]
CAPTURE_DIR = ./caps/
MODEL_DIR = ./model/
TRAINING_DATA_DIR = ./trainings_data/
```

Setup is done.

## Using the tool

The general workflow of the tool consists of 4 steps:

1. Setup
2. Capture data
3. Train models based on the captured data
4. Analyse pcap files

In the following a full example with all steps is given.

### Setup

### Capture Data

Switch to the "./src/python" directory.

```bash
# cd ./src/python/
# sudo python3 esat-capture-data.py
```

Enter a filename for the capture to start the capture process. This will capture all network packages that pass the
interface you specified prior in the esat-config.ini.

The capture process is aborted with: ``CTRL+C``

### Train Model

TODO

### Analyse pcap files
