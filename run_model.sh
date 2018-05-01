#!/bin/bash

#SBATCH -p short   # queue name
#SBATCH -t 0-12:00       # hours:minutes runlimit after which job will be killed.
#SBATCH -n 32      # number of cores requested
#SBATCH --mem=32G # memory requested
#SBATCH -J 707model         # Job name
#SBATCH -o %j.out       # File to which standard out will be written
#SBATCH -e %j.err       # File to which standard err will be written
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Peter_Shen@hms.harvard.edu

# use keras
source ~/keras/bin/activate

python modeling.py
