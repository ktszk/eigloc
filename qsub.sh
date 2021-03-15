#!/bin/sh
#$ -N eigcks
#$ -pe smp 32
#$ -cwd
#$ -V
#$ -q all.q
#$ -S /bin/bash
ncore=1

python dia.py

