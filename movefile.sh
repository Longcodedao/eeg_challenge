#!/bin/bash

ROOT_DIR="MyNeurIPSData/MyNeurIPSData/"

file="3"
temp_folder="ds00550${4+file}-bdf/"
mkdir -p "$temp_folder"

src="$ROOT_DIR/HBN_DATA_FULL/R${file}_L100_bdf/*"
mv $src "$temp_folder"

dst="$ROOT_DIR/HBN_DATA_FULL/R${file}_L100_bdf/$temp_folder"

mv "$temp_folder" "$dst"
rm -rf "$temp_folder"

