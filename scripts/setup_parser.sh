#!/usr/bin/env bash

git clone --branch qanom https://github.com/kleinay/nrl-qasrl.git qanom_parser
# create a softlink from nrl-qasrl repo to `qanom` package
ln -s `pwd`/qanom qanom_parser/qanom
pip install allennlp==0.9.0