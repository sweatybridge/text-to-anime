#!/bin/bash

curl --parallel-max 3 \
-Z 'https://zenodo.org/record/1188976/files/Video_Speech_Actor_[01-24].zip?download=1' \
-o 'ravdess/Speech_Actor_#1.zip'
