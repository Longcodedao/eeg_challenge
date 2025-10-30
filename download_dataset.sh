#!/bin/bash

if ! command -v unzip &> /dev/null
then
    echo "--------------------------------------------------"
    echo "Unzip is not found."
    echo "Installing AWS CLI right now"
    sudo apt-get install unzip
fi


if ! command -v aws &> /dev/null
then
    echo "--------------------------------------------------"
    echo "AWS CLI (command 'aws') is not found."
    echo "Installing AWS CLI right now"
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
    unzip awscliv2.zip
    sudo ./aws/install

    rm -rf awscliv2.zip
    rm -rf ./aws
fi

echo "Starting download the dataset. This may take a while" 

# Create the target directory if it doesn't exist
mkdir -p MyNeurIPSData

# Run the AWS command
aws s3 cp --recursive s3://nmdatasets/NeurIPS25/R1_mini_L100_bdf ./MyNeurIPSData --no-sign-request

echo "Download complete! Files are in the 'MyNeurIPSData' Folder."