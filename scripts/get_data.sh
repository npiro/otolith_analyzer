#!/bin/bash

echo "Downloading data..."

wget https://nimbus.imedea.uib-csic.es/s/8nfnTtfA8z3yMyj/download -O data.zip

unzip data.zip -d data/

rm data.zip
