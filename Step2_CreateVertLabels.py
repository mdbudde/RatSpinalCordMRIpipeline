#!/usr/bin/env python3
"""
==============

nipype routine to iteratively open images to place or modify labels in the volume

"""

import csv
import os  # system functions
import re
import shutil
import subprocess
import glob
from pathlib import Path
import argparse

from SCTCommands import *

imagefile = os.path.join('t2sag','t2sag_echo1.nii')
labelstring = '1,2,3,4,5,6,7,8,9'

thisfilepath = os.path.abspath(os.path.dirname(__file__))
configpath = os.path.abspath(os.path.join(thisfilepath,'config'))

def main():
    #get input arguments
    parser = argparse.ArgumentParser(description='Prints all files from a certain scan.')
    parser.add_argument('-f', type=str, required=False, help='scanlogcsv',default='MRIScanLog.csv')
    parser.add_argument('-r', required=False, help='scanlogcsv', action='store_true')
    args = parser.parse_args()
    scanlogcsv = args.f
    openexisting = args.r

    #open the csv file for each row is a dataset (session/experiment)
    with open(scanlogcsv, newline='') as csvfile:
        print('Reading csv file: ' + scanlogcsv)
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader, None)  # skip the headers, !!!it would be better here to use the dictionary to match up the keys with the header names.


        for row in reader:
            print(', '.join(row))
            subject = row[0]
            session = row[1]

            currsesspath = os.path.join(subject,session)
            print(currsesspath)

            labelsdir = os.path.join(subject,session,'labels')
            if not os.path.isdir(labelsdir):
                os.mkdir(labelsdir)

            labelsinputfile = os.path.join(currsesspath,imagefile)
            labelsoutfile = os.path.join(labelsdir,'labels.nii.gz')
            labelsmeanimage = os.path.join(labelsdir,'imgmeanin.nii.gz')


            if not os.path.exists(labelsoutfile):
                if os.path.exists(labelsinputfile):
                    if not os.path.exists(labelsmeanimage):


                        print('First Image in time (Registration target) ')
                        meancmd = ' fslroi ' + labelsinputfile + ' ' + labelsmeanimage + ' 0 -1 0 -1 0 -1 0 1'
                        print(meancmd)
                        process = subprocess.run(meancmd, stdout=subprocess.PIPE,shell=True)

                    else:
                        print('Mean image already exists')
                else:
                    print('image input not found' + labelsinputfile)

                if os.path.exists(labelsmeanimage):
                        print('Creating labels ' + labelsoutfile )
                        sctcmd = 'sct_label_utils -i ' + labelsmeanimage + ' -create-viewer ' + labelstring + ' -o ' + labelsoutfile
                        print(sctcmd)
                        process = subprocess.run(sctcmd, stdout=subprocess.PIPE,shell=True)
                else:
                    print('img mean input not found')
            else:
                if openexisting:
                    sctcmd = 'sct_label_utils -i ' + labelsmeanimage + ' -create-viewer ' + labelstring + ' -ilabel ' + labelsoutfile + ' -o ' + labelsoutfile
                    print(sctcmd)
                    process = subprocess.run(sctcmd, stdout=subprocess.PIPE,shell=True)
                else:
                    print('Labels already exists. Use -r flag to reopen existing labels in the viewer.')



if __name__ == '__main__':
    main()
