#!/usr/bin/env python
"""
==============
Script for FSL randomize - Two-Group Difference Adjusted for Covariate
Covariates are mean centered within groups in this model by subtracting the group mean from each individual value.
https://afni.nimh.nih.gov/pub/dist/doc/htmldoc/statistics/center.html#when-to-center-within-or-across-groups

MDB and BPM, 10/25/2021

use file AdditionalFiles/MRIScanLog_ABC.csv

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
from nipype.interfaces.utility import Function

import nipype.caching.memory as mem

from SCTCommands import *


# Create custom interface to pass the type/metric for iteratables.
# Get list of filenames from imported parameter/subject list.
# This is done to avoid wildcards and sorting from mixing up the group lists and
# is therefore more explicit than DataGrabber, for example.
from nipype.interfaces.base import BaseInterfaceInputSpec, BaseInterface, File, TraitedSpec, OutputMultiPath, traits




NProcs = 3

nperms = 20


runList = ['Ax'] #'Sag','Ax' for both

#regressors and contrast must match one another
#regressor numbers are from the behav part of the input file.
# in this example, the columns get everything from column 8 and greater (zero is first column!)
# then the regressors or 0 or greater from within that list.
#note that regressor 0 here is a list of ones, so is included in the correlations, effectively the same as de-meaning the data.


regressorlist = [
                # [0, 3],
                #  [0, 3],
                #  [0, 4],
                #  [0, 4],
                #  [0, 7],
                #  [0, 7],
                #  [0, 8],
                #  [0, 8],
                #  [0, 3, 4],
                #  [0, 3, 4],
                #  [0, 5, 6],
                #  [0, 5, 6],
                #  [0, 11],
                #  [0, 11],
                #  [0, 12],
                #  [0, 12],
                 [0, 3, 13],
                 [0, 3, 13],
                 [0, 4, 13],
                 [0, 4, 13],
                  [0, 5, 13],
                  [0, 5, 13],
                  [0, 6, 13],
                  [0, 7, 13]]

contrastlist = [
                # [0, 1],
                # [0, -1],
                # [0, 1],
                # [0, -1],
                # [0, 1],
                # [0, -1],
                # [0, 1],
                # [0, -1],
                # [0, 0, 1],
                # [0, 0, -1],
                # [0, 0, 1],
                # [0, 0, -1],
                # [0, 1],
                # [0, -1],
                # [0, 1],
                # [0, -1],
                [0, 1, 0],
                [0, -1, 0],
                [0, 1, 0],
                [0, -1, 0],
                [0, 1, 0],
                [0, -1, 0],
                [0, 1, 0],
                [0, -1, 0]]



class GenFileListInputSpec(BaseInterfaceInputSpec):
    base_dir = traits.Str(mandatory=False, desc='base dir')
    type = traits.Str(mandatory=True, desc='type')
    metric = traits.Str(mandatory=True, desc='metric')
    randHack = traits.Str(mandatory=False, desc='rand')

class GenFileListOutputSpec(TraitedSpec):
    outfilelist = OutputMultiPath(File(exists=True))

class GenFileList(BaseInterface):
    input_spec = GenFileListInputSpec
    output_spec = GenFileListOutputSpec

    def _run_interface(self, runtime):

        print('Base is: '+str(self.inputs.base_dir))
        print('Type is: '+str(self.inputs.type)+'  Metric is: '+str(self.inputs.metric) )
        # Call our python code here:
        #result = getlist.run()
        # And we are done
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['outfilelist'] = GenerateInputFileList(
            str(self.inputs.base_dir),
            str(self.inputs.type),
            str(self.inputs.metric)
        )
        return outputs


class GenModelsInputSpec(BaseInterfaceInputSpec):
    regressors = traits.List(traits.List(traits.Float(mandatory=False, desc='regressors')))
    regressorlist = traits.List(traits.Int(mandatory=False, desc='regressorlist'))
    contrastlist = traits.List(traits.Int(mandatory=False, desc='contrasts'))
    randHack = traits.Str(mandatory=False, desc='rand')

class GenModelsOutputSpec(TraitedSpec):
    mat_file = OutputMultiPath(File(desc='matfile'))
    con_file = OutputMultiPath(File(desc='confile'))

class GenModels(BaseInterface):
    input_spec = GenModelsInputSpec
    output_spec = GenModelsOutputSpec

    def _run_interface(self, runtime):


        # Call our python code here:
        #result = getlist.run()
        # And we are done
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['mat_file'], outputs['con_file'] = GenerateModels(
            self.inputs.regressors,
            self.inputs.regressorlist,
            self.inputs.contrastlist
        )
        return outputs






#some generic settings that don't need modification
thisfilepath = os.path.abspath(os.path.dirname(__file__))
configpath = os.path.abspath(os.path.join(thisfilepath,'config'))
templatepath = os.path.abspath(os.path.join(thisfilepath, 'Templates/RatHistoAtlas/template/'))
#t2template  = os.path.abspath(os.path.join(thisfilepath,'Templates','HarrisAtlas2020Update68GM18WM','T2Brain_ReorientResizeSmoothed_p3xp3xp3.nii.gz'))
#t2brainmask = os.path.abspath(os.path.join(thisfilepath,'Templates','HarrisAtlas2020Update68GM18WM','T2Brain_ReorientResizeSmoothedMasked_p3xp3xp3.nii.gz'))


def getFullProcessingList():
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
        sexlist = []
        injtypelist = []
        behavlist = []
        for row in reader:
            print(', '.join(row))
            subject = row[0]
            session = row[1]
            injtype = row[7]
            sex = row[8]
            behav = row[8:]
            behavfloat = [float(item) for item in behav]
            subjectlist.append(subject)
            sessionlist.append(session)
            injtypelist.append(injtype)
            sexlist.append(sex)
            behavlist.append(behavfloat)
    return subjectlist, sessionlist, sexlist, injtypelist, behavlist



def GenerateInputFileList(basedir, type, metric):

    metricfilelist = []
    fcount = 0
    time = '1D'
    for curnum in range(len(subject_list)):
        currpath = basedir+'/templatespace/'+type+'/'+subject_list[curnum]+'_' + time + '_'+subject_list[curnum]+'/*'+metric+'*.nii.gz'
        print('     Search string is: '+currpath)
        currstr = glob.glob(currpath)
        if not len(currstr) == 0:
            print(currstr)
            metricfilelist.append(currstr[0])
            fcount += 1

    #print(metricfilelist)
    return metricfilelist



def GenerateModels(regressors, regressorlist, contrastlist):

    # print('Regressors')
    # print(type(regressors))
    # print(regressors)
    # print('RegressorList')
    # print(type(regressorlist))
    # print(regressorlist)
    # print('ContrastList')
    # print(type(contrastlist))
    # print(contrastlist)

    regressorlist = [regressorlist]
    contrastlist = [contrastlist]

    regnrows = len(regressors)
    nconts = len(contrastlist)

    matfilenames = []
    confilenames = []
    ncurreg = 0

    for curreg in regressorlist:
        nregs = len(curreg)

        matfilename = os.path.abspath('randomise'+str(ncurreg)+'.mat')
        print('Writing '+matfilename)
        with open(matfilename,'w') as matfile:
            matfile.write('/NumWaves ' + str(nregs)+' \n')
            matfile.write('/NumPoints ' + str(regnrows) + '\n')
            matfile.write('/Matrix\n')

            for regline in range(regnrows):
                matlist = []
                for regcol in curreg:

                    currVal = regressors[regline][regcol]
                    matlist.append(currVal)

                matstr = " ".join([str(elem) for elem in matlist]) + ' \n'
                matfile.write(matstr)
        matfilenames.append(matfilename)

        with open(matfilename,'r') as matfile:
            for line in matfile:
                print(line, end = '')

        ncurreg += 1

    ncurcon = 0
    for curcon in contrastlist:
        confilename = os.path.abspath('randomise'+str(ncurcon)+'.con')
        print('Writing '+confilename)
        with open(confilename,'w') as confile:
            confile.write('/NumWaves '+ str(nregs)+' \n')
            confile.write('/NumContrasts 1\n')
            confile.write('/Matrix \n')


            constr = " ".join([str(elem) for elem in curcon]) + ' \n'
            confile.write(constr)

        confilenames.append(confilename)


        with open(confilename,'r') as confile:
            for line in confile:
                print(line, end = '')

        ncurcon += 1

        #end of list of regressors
    return matfilenames, confilenames

"""
Get All of the processing lists
"""

subject_list,session_list,sexlist,injtypelist,behavlist = getFullProcessingList()

print(behavlist)


"""
Map field names to individual subject runs
"""


metric_listSag = ['T2Map_Sag_reg']

type_listSag = ['t2']


statsSag = pe.Workflow(name="statsSag")
statsSag.base_dir = os.path.abspath('.')

infosourceSag = pe.Node(
    interface=util.IdentityInterface(fields=['type_list','metric_list']), name="infosourceSag")
infosourceSag.iterables = ([('type_list', type_listSag),('metric_list', metric_listSag)])
infosourceSag.synchronize = True

#
# corrcol = pe.Node(
#     interface=util.IdentityInterface(fields=['column2corr']), name="corrcol")
# corrcol.iterables = ('column2corr', [0, 1])
#


filelistinterface = pe.Node(interface=GenFileList(), name='filelistinterface')
filelistinterface.inputs.base_dir = statsSag.base_dir
statsSag.connect(infosourceSag,'type_list',  filelistinterface,'type')
statsSag.connect(infosourceSag,'metric_list',filelistinterface,'metric')
#the following is a hack to rerun the file listing for each run. It passes a random number so the input are always out of date.
filelistinterface.inputs.randHack = str(random.random())


mergedfiles = pe.Node(interface=fsl.Merge(), name='mergedfiles')
mergedfiles.inputs.dimension = 't'
statsSag.connect(filelistinterface, 'outfilelist', mergedfiles, 'in_files')

sagcrop = pe.Node(interface=SCTCrop(), name='sagcrop')
sagcrop.inputs.ymin = "0"
sagcrop.inputs.ymax = "-1"
sagcrop.inputs.xmin = "94"
sagcrop.inputs.xmax = "97"
sagcrop.inputs.zmin = "0"
sagcrop.inputs.zmax = "-1"
statsSag.connect(mergedfiles, 'merged_file', sagcrop, 'in_file')

allmean = pe.Node(interface=fsl.MeanImage(), name='allmean')
allmean.inputs.dimension = 'T'
statsSag.connect(sagcrop, 'out_file', allmean, 'in_file')


regMask_crop = pe.Node(interface=SCTCrop(), name='regMask_crop')
regMask_crop.inputs.ymin = "0"
regMask_crop.inputs.ymax = "-1"
regMask_crop.inputs.xmin = "94"
regMask_crop.inputs.xmax = "97"
regMask_crop.inputs.zmin = "428"
regMask_crop.inputs.zmax = "559"
regMask_crop.inputs.in_file = templatepath + '/RatHistoAtlas_regMask.nii.gz'





randomize = pe.Node(interface=fsl.Randomise(), name='randomize')
randomize.inputs.tfce = True
#randomize.inputs.tfce2D = True
randomize.inputs.vox_p_values = True
randomize.inputs.num_perm = nperms
#randomize.inputs.demean = True
#randomize.inputs.mask = t2brainmask
statsSag.connect(sagcrop, 'out_file', randomize, 'in_file')
# statsSag.connect(filelistinterface, 'mat_file', randomize, 'design_mat')
# statsSag.connect(filelistinterface, 'con_file', randomize, 'tcon')
statsSag.connect(regMask_crop, 'out_file', randomize, 'mask')




metric_listAx = ['DTI_Daxial_reg','DDE_Daxial_reg','T2Map_Axial_reg','DTI_FA_reg','DTI_MD_reg']

type_listAx = ['dti','dde','t2','dti','dti']


statsAx = pe.Workflow(name="statsAx")
statsAx.base_dir = os.path.abspath('.')

infosourceAx = pe.Node(
    interface=util.IdentityInterface(fields=['type_list','metric_list']), name="infosourceAx")
infosourceAx.iterables = ([('type_list', type_listAx),('metric_list', metric_listAx),])
infosourceAx.synchronize = True



filelistinterfaceAx = pe.Node(interface=GenFileList(), name='filelistinterfaceAx')
filelistinterfaceAx.inputs.base_dir = statsAx.base_dir
statsAx.connect(infosourceAx,'type_list',  filelistinterfaceAx,'type')
statsAx.connect(infosourceAx,'metric_list',filelistinterfaceAx,'metric')
#the following is a hack to rerun the file listing for each run. It passes a random number so the input are always out of date.
filelistinterfaceAx.inputs.randHack = str(random.random())



mergedfilesAx = pe.Node(interface=fsl.Merge(), name='mergedfilesAx')
mergedfilesAx.inputs.dimension = 't'
statsAx.connect(filelistinterfaceAx, 'outfilelist', mergedfilesAx, 'in_files')


axcrop = pe.Node(interface=SCTCrop(), name='axcrop')
axcrop.inputs.ymin = "0"
axcrop.inputs.ymax = "-1"
axcrop.inputs.xmin = "0"
axcrop.inputs.xmax = "-1"
axcrop.inputs.zmin = "26"
axcrop.inputs.zmax = "90"
statsAx.connect(mergedfilesAx, 'merged_file', axcrop, 'in_file')


axregMask_crop = pe.Node(interface=SCTCrop(), name='axregMask_crop')
axregMask_crop.inputs.ymin = "0"
axregMask_crop.inputs.ymax = "-1"
axregMask_crop.inputs.xmin = "0"
axregMask_crop.inputs.xmax = "-1"
axregMask_crop.inputs.zmin = "454" #original reg crop 428 + above 26
axregMask_crop.inputs.zmax = "518" #454 + 64 slices in above crop
axregMask_crop.inputs.in_file = templatepath + '/RatHistoAtlas_regMask.nii.gz'



#This is an interface to a local node to generate mat/con files base on the listed regressors and contrasts.
# it is effectively a clean way to iterate across a large number of different stats models based on a single
#input file with many different regressors/contrasts that one would like to examine.

modelinterfaceAx = pe.Node(interface=GenModels(), name='modelinterfaceAx')
modelinterfaceAx.inputs.regressors = behavlist
modelinterfaceAx.inputs.randHack = str(random.random())

modelinterfaceAx.iterables = ([('regressorlist', regressorlist),('contrastlist', contrastlist)])
modelinterfaceAx.synchronize = True


randomizeAx = pe.Node(interface=fsl.Randomise(), name='randomizeAx')
randomizeAx.inputs.tfce = True
randomizeAx.inputs.vox_p_values = True
randomizeAx.inputs.num_perm = nperms
#randomize.inputs.demean = True
#randomize.inputs.mask = t2brainmask
statsAx.connect(axcrop, 'out_file', randomizeAx, 'in_file')
statsAx.connect(modelinterfaceAx, 'mat_file', randomizeAx, 'design_mat')
statsAx.connect(modelinterfaceAx, 'con_file', randomizeAx, 'tcon')
statsAx.connect(axregMask_crop, 'out_file', randomizeAx, 'mask')



sl = pe.Node(interface=util.Select(), name='sl')
sl.inputs.index = [0]
statsAx.connect(randomizeAx, 'tstat_files', sl, 'inlist')

tsqr = pe.Node(interface=fsl.UnaryMaths(), name='tsqr')
tsqr.inputs.operation = "sqr"
statsAx.connect(sl, 'out', tsqr, 'in_file')

tsqrDF = pe.Node(interface=fsl.maths.MathsCommand(), name='tsqrDF')
tsqrDF.inputs.args = "-add 38"
statsAx.connect(tsqr, 'out_file', tsqrDF, 'in_file')

Rstat = pe.Node(interface=fsl.BinaryMaths(), name='Rstat')
Rstat.inputs.operation = "div"
Rstat.inputs.args = "-sqrt"
statsAx.connect(tsqr, 'out_file', Rstat, 'in_file')
statsAx.connect(tsqrDF, 'out_file', Rstat, 'operand_file')


tsign = pe.Node(interface=fsl.maths.MathsCommand(), name='tsign')
tsign.inputs.args = "-thr 0 -bin -mul 2 -sub 1 "
statsAx.connect(sl, 'out', tsign, 'in_file')

Rstatsigned = pe.Node(interface=fsl.BinaryMaths(), name='Rstatsigned')
Rstatsigned.inputs.operation = "mul"
statsAx.connect(Rstat, 'out_file', Rstatsigned, 'in_file')
statsAx.connect(tsign, 'out_file', Rstatsigned, 'operand_file')



"""
End of preprocessing nodes
"""


"""
Setup the datasink/storage location of produced maps
"""

datasink = pe.Node(interface=nio.DataSink(), name='datasink')
datasink.inputs.base_directory = os.path.abspath('results')

datasink.inputs.substitutions = [('_subject_id_', '_'),('_session_id_', '')]



# f_corrected_p_files (a list of items which are a pathlike object or string representing an existing file) – F contrast FWE (Family-wise error) corrected p values files.
# f_p_files (a list of items which are a pathlike object or string representing an existing file) – F contrast uncorrected p values files.
# fstat_files (a list of items which are a pathlike object or string representing an existing file) – F contrast raw statistic.
# t_corrected_p_files (a list of items which are a pathlike object or string representing an existing file) – T contrast FWE (Family-wise error) corrected p values files.
# t_p_files (a list of items which are a pathlike object or string representing an existing file) – F contrast uncorrected p values files.
# tstat_files (a list of items which are a pathlike object or string representing an existing file) – T contrast raw statistic.

statsAx.connect(Rstatsigned,'out_file',datasink,'Rsquared')
# register.connect(t2sag_reg,'warp_file',datasink,'t2.@warp')
statsAx.connect(randomizeAx,'t_corrected_p_files',datasink,'corrp')
statsAx.connect(randomizeAx,'tstat_files',datasink,'tstat')




if __name__ == '__main__':
    #print('run')
    if 'Sag' in runList:
        statsSag.run()

    if 'Ax' in runList:
        #statsAx.run(plugin='MultiProc', plugin_args={'n_procs' : NProcs})
        statsAx.run()
