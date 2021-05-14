#!/bin/bash

module load Python

export WORKON_HOME=$HOME/.virtualenvs
export VIRTUALENVWRAPPER_PYTHON=/usr/local/Cluster-software/software/Python/3.8.6-GCCcore-10.2.0/bin/python3
export VIRTUALENVWRAPPER_VIRTUALENV=/usr/local/Cluster-software/software/Python/3.8.6-GCCcore-10.2.0/bin/virtualenv
source /usr/local/Cluster-software/software/Python/3.8.6-GCCcore-10.2.0/bin/virtualenvwrapper.sh

workon evobcienv

#$ -cwd
#$ -t 1-31
#$ -N a-2
#$ -e $HOME/error/evobci/
#$ -o $HOME/output/evobci/

python3 $HOME/evobci/experiment.py -ps 50 -ng 200 -mp 10.0 -md 1.0 200 $HOME/evobci/input/A-2.csv $HOME