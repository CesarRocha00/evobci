#!/bin/bash
module load Python

export WORKON_HOME=$HOME/.virtualenvs
export VIRTUALENVWRAPPER_PYTHON=/usr/local/Cluster-software/software/Python/3.8.6-GCCcore-10.2.0/bin/python3
export VIRTUALENVWRAPPER_VIRTUALENV=/usr/local/Cluster-software/software/Python/3.8.6-GCCcore-10.2.0/bin/virtualenv
source /usr/local/Cluster-software/software/Python/3.8.6-GCCcore-10.2.0/bin/virtualenvwrapper.sh

workon evobcienv

qsub $HOME/evobci/script/eeg_cc_1.sh
qsub $HOME/evobci/script/eeg_cc_2.sh
qsub $HOME/evobci/script/eeg_nr_1.sh
qsub $HOME/evobci/script/eeg_nr_2.sh