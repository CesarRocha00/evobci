#!/bin/bash
#$ -t 1-31
#$ -N eeg_nr_1
#$ -cwd
#$ -e $HOME/error/
#$ -o $HOME/output/

python3 $HOME/evobci/experiment.py -ps 50 -ng 200 -mp 10.0 -md 1.0 200 $HOME/evobci/input/NR_1.csv $HOME