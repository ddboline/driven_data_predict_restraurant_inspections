#!/bin/bash

# sudo apt-get update
# sudo apt-get install -y ipython python-matplotlib python-sklearn python-pandas htop

# sudo cp -a ~/.ssh /root/
# sudo bash -c "echo deb ssh://ddboline@ddbolineathome.mooo.com/var/www/html/deb/trusty/devel ./ > /etc/apt/sources.list.d/py2deb2.list"
# sudo apt-get update
# 
# sudo apt-get install -y --force-yes xgboost python-xgboost ipython python-matplotlib python-scikit-learn python-pandas htop

### a horrible hack...
sudo cp -a ~/.ssh /root/
sudo bash -c "echo deb ssh://ddboline@ddbolineathome.mooo.com/var/www/html/deb/trusty/devel ./ > /etc/apt/sources.list.d/py2deb2.list"
sudo apt-get update

sudo apt-get install -y --force-yes python-nltk python-gensim

CURDIR=`pwd`
cd $HOME
scp ddboline@ddbolineathome.mooo.com:~/nltk_data_full.tar.gz .
tar zxvf nltk_data_full.tar.gz
rm nltk_data_full.tar.gz
cd $CURDIR
