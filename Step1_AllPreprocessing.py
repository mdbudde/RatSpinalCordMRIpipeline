#!/usr/bin/env python
"""
==============
All  preprocessing pipeline
==============

Assumes data inputs of:
    ie file phase: interleaved label control image in phase format
        (real label, real control, imag label, imag control)

        Uses Matlab nipype interface to process the images

"""

"""
Tell python where to find the appropriate functions.
"""
 
import os  # system functions
import argparse
import csv
import glob
import random


import nipype.interfaces.io as nio  # Data i/o
import nipype.interfaces.fsl as fsl  # fsl
import nipype.interfaces.utility as util  # utility
import nipype.pipeline.engine as pe  # pypeline engine
from nipype.interfaces.utility import Rename
from niflow.nipype1.workflows import *
from nipype import Function

#local wrappers for matlab specific processing scripts
from DWIMapsMatlab import DWIMapsMatlab
from T2MapMatlab import T2MapMatlab

from CustomLocalInterfaces import *

#main matlab path (on mac, the .app directory)
matlabs = glob.glob("/Applications/MATLAB_R*.app")
matlabpath = matlabs[-1] #get the last instance of the matlab, which should return the newest version alphabetically
print('Matlab path ' + matlabpath)



#starting in same folder as this file
qmrilabpath = os.path.abspath(os.path.dirname(__file__)) + '/MatlabTools/qMRLab'


#some generic settings that don't need modification
thisfilepath = os.path.abspath(os.path.dirname(__file__))
configpath = os.path.abspath(os.path.join(thisfilepath,'config'))
matlabtoolspath = os.path.abspath(os.path.join(thisfilepath,'MatlabTools'))

"""
Map field names to individual subject runs
"""

def getProcessingList():
    #get input arguments
    parser = argparse.ArgumentParser(description='Prints all files from a certain scan.')
    parser.add_argument('-f', type=str, required=False, help='scanlogcsv',default='MRIScanLog.csv')
    args = parser.parse_args()
    scanlogcsv = args.f

    #open the csv file for each row is a dataset (session/experiment)
    with open(scanlogcsv, newline='') as csvfile:
        print('Reading csv file: ' + scanlogcsv)
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader, None)  # skip the headers, !!!it would be better here to use the dictionary to match up the keys with the header names.

        subjectlist = []
        sessionlist = []
        for row in reader:
            print(', '.join(row))
            subject = row[0]
            session = row[1]
            subjectlist.append(subject)
            sessionlist.append(session)
    return subjectlist, sessionlist


subject_list,session_list = getProcessingList()


info = dict(
    dti_file=[['subject_id', 'session_id', 'dti', 'dti']],  #subject/subject_session/ie/ie_phase.nii
    dde_file=[['subject_id', 'session_id', 'dde', 'dde']],  #subject/subject_session/ie/ie_phase.nii
    t2sag_files=[['subject_id', 'session_id', 't2sag', 't2sag']],  #subject/subject_session/ie/ie_phase.nii
    t2ax_files=[['subject_id', 'session_id', 't2ax', 't2ax']],  #subject/subject_session/ie/ie_phase.nii
    mge_files=[['subject_id', 'session_id', 'mge', 'mge_echo']]  #subject/subject_session/ie/ie_phase.nii
    )

datasource = pe.Node(
    interface=nio.DataGrabber(
        infields=['subject_id','session_id'], outfields=list(info.keys())),
    name='datasource')
datasource.inputs.template = "%s_%s/%s/%s"
datasource.inputs.base_directory = os.path.abspath('.')
datasource.inputs.field_template = dict(
        dti_file='%s/%s/%s/%s.nii',
        dde_file='%s/%s/%s/%s.nii',
        t2sag_files='%s/%s/%s/%s*.nii',
        t2ax_files='%s/%s/%s/%s*.nii',
        mge_files='%s/%s/%s/%s*.nii'
        )
datasource.inputs.template_args = info
datasource.inputs.sort_filelist = True

"""
Inputnode passes the files that are used in the processing, but true filename provided from datasource/datagrabber
"""
inputnode = pe.Node(
    interface=util.IdentityInterface(fields=[
        'dti_file'
    ]),
    name='inputspec')


infosource = pe.Node(
    interface=util.IdentityInterface(fields=['subject_id','session_id']), name="infosource")

infosource.iterables = ([('session_id', session_list),('subject_id', subject_list)])
infosource.synchronize = True

"""
Specify the main workflow, staring with
    infosource(subjects/sessions) -> datasource(files) -> inputnode(available to subsequent steps)
"""


preproc = pe.Workflow(name="preproc")
preproc.base_dir = os.path.abspath('.')

#subject/session connectors
preproc.connect(infosource, 'subject_id', datasource, 'subject_id')
preproc.connect(infosource, 'session_id', datasource, 'session_id')

#data file connectors
preproc.connect(datasource, 'dti_file', inputnode, 'dti_file')
preproc.connect(datasource, 'dde_file', inputnode, 'dde_file')
preproc.connect(datasource, 't2sag_files', inputnode, 't2sag_files')
preproc.connect(datasource, 't2ax_files', inputnode, 't2ax_files')
preproc.connect(datasource, 'mge_files', inputnode, 'mge_files')


"""
DTI
"""
mlabDTI = pe.Node(interface=DWIMapsMatlab(),name='mlabDTI')
#assumes the matlab file is in the same directory as this script, hence __file__ is the current file
mlabDTI.inputs.script_file = thisfilepath + '/DWI_fADCmap.m'
mlabDTI.inputs.bvecs_file = os.path.join(configpath,'dti.bvecs')
mlabDTI.inputs.bvals_file = os.path.join(configpath,'dti.bvals')
mlabDTI.inputs.omitindices = '[]' #matlab syntax
mlabDTI.inputs.b0indices = '[1 2 3]' #matlab syntax
mlabDTI.inputs.dwimode = 0 #DWI Mode, see matlab code: 1=2dDWI (spinal cord optimized)
mlabDTI.inputs.matlabpath = matlabpath
mlabDTI.inputs.localmatlabpath = matlabtoolspath
preproc.connect(inputnode,'dti_file',mlabDTI,'dwi_file')


"""
fDWI, single axis
"""
mlabDfWI = pe.Node(interface=DWIMapsMatlab(),name='mlabDfWI')
#assumes the matlab file is in the same directory as this script, hence __file__ is the current file
mlabDfWI.inputs.script_file = thisfilepath + '/DWI_fADCmap.m'
mlabDfWI.inputs.bvecs_file = os.path.join(configpath,'fdwi.bvecs')
mlabDfWI.inputs.bvals_file = os.path.join(configpath,'fdwi.bvals')
mlabDfWI.inputs.omitindices = '[1 2 3 4 5 16 17 18]' #matlab syntax
mlabDfWI.inputs.b0indices = '[6 11 19 24]' #matlab syntax
mlabDfWI.inputs.dwimode = 2 #DWI Mode, see matlab code: 2=1dDWI (spinal cord optimized)
mlabDfWI.inputs.matlabpath = matlabpath
mlabDfWI.inputs.localmatlabpath = matlabtoolspath
preproc.connect(inputnode,'dde_file',mlabDfWI,'dwi_file')


"""
T2 Sagittal maps
"""
mlabT2sag = pe.Node(interface=T2MapMatlab(),name='mlabT2sag')
#assumes the matlab file is in the same directory as this script, hence __file__ is the current file
mlabT2sag.inputs.script_file = thisfilepath + '/T2map.m'
mlabT2sag.inputs.tetimes_file = os.path.join(configpath,'t2sag.tetimes')
mlabT2sag.inputs.matlabpath = matlabpath
mlabT2sag.inputs.localmatlabpath = matlabtoolspath
preproc.connect(inputnode,'t2sag_files',mlabT2sag,'in_files')

"""
T2 Axial maps
"""
mlabT2ax = pe.Node(interface=T2MapMatlab(),name='mlabT2ax')
#assumes the matlab file is in the same directory as this script, hence __file__ is the current file
mlabT2ax.inputs.script_file = thisfilepath + '/T2map.m'
mlabT2ax.inputs.tetimes_file = os.path.join(configpath,'t2ax.tetimes')
mlabT2ax.inputs.matlabpath = matlabpath
mlabT2ax.inputs.localmatlabpath = matlabtoolspath
preproc.connect(inputnode,'t2ax_files',mlabT2ax,'in_files')

"""
T2* 3D maps
"""
#mlabT2star = pe.Node(interface=T2MapMatlab(),name='mlabT2star')
#assumes the matlab file is in the same directory as this script, hence __file__ is the current file
#mlabT2star.inputs.script_file = thisfilepath + '/T2map.m'
#mlabT2star.inputs.tetimes_file = os.path.join(configpath,'t2star.tetimes')
#mlabT2star.inputs.matlabpath = matlabpath
#mlabT2star.inputs.localmatlabpath = matlabtoolspath
#preproc.connect(inputnode,'mge_files',mlabT2star,'in_files')

"""
MGE mean across time
"""
mgeMerge = pe.Node(interface=fsl.Merge(),name='mgeMerge')
mgeMerge.inputs.dimension = 't'
preproc.connect(inputnode,'mge_files',mgeMerge,'in_files')

mgeMean = pe.Node(interface=fsl.ImageMaths(),name='mgeMean')
mgeMean.inputs.args = '-Tmean'
preproc.connect(mgeMerge,'merged_file',mgeMean,'in_file')

"""
End of preprocessing nodes
"""


"""
Connect all processed files to the derivatives output location
"""
datasink = pe.Node(interface=nio.DataSink(), name='datasink')
datasink.inputs.base_directory = os.path.abspath('derivatives')
datasink.inputs.substitutions = [('_subject_id_', '_'),('_session_id_', '')]

preproc.connect(mlabDTI,'fa_file',datasink,'dti')
preproc.connect(mlabDTI,'md_file',datasink,'dti.@md')
preproc.connect(mlabDTI,'fadc_file',datasink,'dti.@l1')

preproc.connect(mlabDfWI,'fadc_file',datasink,'dde')
preproc.connect(mlabDfWI,'md_file',datasink,'dde.@md')
preproc.connect(mlabDfWI,'meanrad_file',datasink,'dde.@mrad')
preproc.connect(mlabDfWI,'meanb0_file',datasink,'dde.@mb0')

preproc.connect(mlabT2sag,'t2map_file',datasink,'t2')
preproc.connect(mlabT2ax,'t2map_file',datasink,'t2.@ax')

preproc.connect(mgeMean,'out_file',datasink,'mge')
#preproc.connect(mlabT2star,'t2map_file',datasink,'mge.@t2star')

if __name__ == '__main__':
    preproc.run()
    #dwiproc.write_graph()
