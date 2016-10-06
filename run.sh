#!/bin/bash

if ! type "clang" > /dev/null; then
	sudo apt-get install clang
	echo "no clang"
fi

if ! type "pip" > /dev/null; then
	sudo apt-get install python-pip
fi

if ! type "virtualenv" > /dev/null; then
	sudo apt-get install virtualenv
fi

virtualenv env

source env/bin/activate

pip install clang
pip install -r req.txt
