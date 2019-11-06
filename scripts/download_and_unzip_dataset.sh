#!/usr/bin/env bash

readonly DOWNLOAD_URL="https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic-signs-data.zip"
readonly FILE_NAME="traffic-signs-data.zip"
readonly CURRENT_DIR=$(dirname "$(readlink -f "$0")")
readonly DATA_DIR=$(realpath "$CURRENT_DIR/../data/")

if [ ! -f $DATA_DIR/$FILE_NAME ]; then
    wget -P $DATA_DIR $DOWNLOAD_URL
fi

if [ ! -f $DATA_DIR/$FILE_NAME ]; then
    echo "Cannot download data"
    exit
fi

cd $DATA_DIR
pwd
unzip $FILE_NAME
echo "Finish unzipping"
cd $CURRENT_DIR
