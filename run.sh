#!/bin/bash

if ! type "clang" > /dev/null; then
	sudo apt -y install clang
	echo "no clang"
fi

if ! type "pip" > /dev/null; then
	sudo apt -y install python-pip
fi

if ! type "virtualenv" > /dev/null; then
	sudo apt -y install virtualenv
fi

virtualenv env

source env/bin/activate

pip install clang
pip install -r req.txt
