#!/usr/bin/env python
"""
==============
Interface used to spatially align data to a rat spinal cord template.

Very briefly, it:
    using the sct_register_to_template:
    - straightens the spinal cord using intra-spinal labels at each vertebral level.
    - aligns these to corresponding labels in the rat spinal cord template.
    using sct_register_multimodal:
    - initializes the t2 weighted image using the above registration parameters (warp) and
        further warps the image to the template using image intensities.
    - does the same for each of the other contrasts. Noting in this example one of the diffusion weighted images (mean filter image from dde)
        has the best SNR and wm/gm contrast, so it is use for all EPI-based registration.

    Finally does some cropping to the cervical cord level since the template includes the whole cervical to lumbar cord.



Note, the GenUnrotateMatrix is the least intuitive feature of this registration process.
In animal spinal cord MRI, often the animal can be twisted or not well-aligned in either the prone
or supine position.  It is also really difficult to automate in-plane rotations and/or slice-by-slice rotations (torsion)
since the spinal cord anatomy has somewhat limited features to promote that alignment (unlike the brain for example).
To provide a solution, we exploit the fact that the scanner operator will align the sagittal images perpendicular to the cord. The GenUnrotateMatrix
interface obtains the rotation matix of that scan relative to the normal x,y,z coordinate system of the magnet itself. The in-plane component of that
rotation matrix is used to 'unrotate' all of the images from that scan.  Effectively, this ensures that the dorsal-ventral
aspect of the cord points along the y axis (up-down) as a starting point.  All other registration parameters start after that step and basically
use the parameters of the spinal cord toolbox.

NOTE: We had to modify the sct_register_to_template file directly to sample at 0.1x0.1x0.1 instead of the default 1x1x1mm used for humans.

"""

"""
Tell python where to find the appropriate functions.
"""

import os  # system functions
import argparse
import csv
import random

import nipype.interfaces.io as nio  # Data i/o
import FSL.fsl as fsl  # fsl
import nipype.interfaces.utility as util  # utility
import nipype.pipeline.engine as pe  # pypeline engine
from nipype.interfaces.utility import Rename
from niflow.nipype1.workflows import *


from SCTCommands import *

from CustomLocalInterfaces import *

redoUnrot = True
if redoUnrot:
    rotval = str(random.random())
else:
    rotval = '123'

thisfilepath = os.path.abspath(os.path.dirname(__file__))
configpath = os.path.abspath(os.path.join(thisfilepath,'config'))
templatepath = os.path.abspath(os.path.join(thisfilepath, 'Templates/RatHistoAtlas/template/'))

currentpath = os.path.abspath('.')


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


#function to get one column of the csv file dynamically within the pipeline
# this is probably not the best way, but functions in nipype are closed environments,
# so the easiest way to get this to work was to re-read it each time.
def getRotInfo(session_id):
    import argparse
    import csv
    #get input arguments
    parser = argparse.ArgumentParser(
            description='Prints all files from a certain scan.')
    parser.add_argument('-f', type=str, required=False,
            help='scanlogcsv', default='MRIScanLog.csv')
    args = parser.parse_args()
    scanlogcsv = args.f

    #open the csv file for each row is a dataset (session/experiment)
    with open(scanlogcsv, newline='') as csvfile:
        #print('Reading csv file: ' + scanlogcsv)
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        # skip the headers, !!!it would be better here to use the dictionary to match up the keys with the header names.
        next(reader, None)

        for row in reader:
            subject = row[0]
            session = row[1]

            if (session == session_id):
                rot = row[7]
                if ((rot == True) | (rot == "True") | (rot == "TRUE") | (rot == 1)):
                    rot = True
                else:
                    rot = False
                return rot
    return False


subject_list,session_list = getProcessingList()

info = dict(
    dti_Daxial_file=[['dti','session_id', 'subject_id','dti_Daxial']],
    dti_fa_file=[['dti','session_id', 'subject_id','dti_FA']],
    dti_md_file=[['dti','session_id', 'subject_id','dti_MD']],
    dde_Daxial_file=[['dde','session_id', 'subject_id','dde_Daxial']],
    t2sag_T2map_file=[['t2','session_id', 'subject_id','t2sag_echo1_T2map']],
    t2ax_T2map_file=[['t2','session_id', 'subject_id','t2ax_echo1_T2map']],
    t2sagorig_file=[['subject_id', 'session_id', 't2sag', 't2sag_echo2']],
    t2axorig_file=[['subject_id', 'session_id', 't2ax', 't2ax_echo2']],
    labels_file=[['subject_id', 'session_id', 'labels', 'labels']],
    dde_MeanB0_file=[['dde','session_id', 'subject_id','dde_MeanB0']]
    )

datasource = pe.Node(
    interface=nio.DataGrabber(
        infields=['subject_id','session_id'], outfields=list(info.keys())),
    name='datasource')
datasource.inputs.template = "derivatives/%s/%s_%s/%s"
datasource.inputs.base_directory = os.path.abspath('.')
datasource.inputs.field_template = dict(
        dti_Daxial_file='derivatives/%s/%s_%s/%s.nii',
        dti_fa_file='derivatives/%s/%s_%s/%s.nii',
        dti_md_file='derivatives/%s/%s_%s/%s.nii',
        dde_Daxial_file='derivatives/%s/%s_%s/%s.nii',
        t2ax_T2map_file='derivatives/%s/%s_%s/%s.nii',
        t2sag_T2map_file='derivatives/%s/%s_%s/%s.nii',
        t2sagorig_file='%s/%s/%s/%s.nii',
        t2axorig_file='%s/%s/%s/%s.nii',
        labels_file='%s/%s/%s/%s.nii.gz',
        dde_MeanB0_file='derivatives/%s/%s_%s/%s.nii'
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


register = pe.Workflow(name="register")
register.base_dir = os.path.abspath('.')

#subject/session connectors
register.connect(infosource, 'subject_id', datasource, 'subject_id')
register.connect(infosource, 'session_id', datasource, 'session_id')

#data file connectors
register.connect(datasource, 't2sag_T2map_file', inputnode, 't2sag_T2map_file')
register.connect(datasource, 't2ax_T2map_file', inputnode, 't2ax_T2map_file')
register.connect(datasource, 'dti_Daxial_file', inputnode, 'dti_Daxial_file')
register.connect(datasource, 'dti_fa_file', inputnode, 'dti_fa_file')
register.connect(datasource, 'dti_md_file', inputnode, 'dti_md_file')
register.connect(datasource, 'dde_Daxial_file', inputnode, 'dde_Daxial_file')
register.connect(datasource, 't2sagorig_file', inputnode, 't2sagorig_file')
register.connect(datasource, 't2axorig_file', inputnode, 't2axorig_file')
register.connect(datasource, 'labels_file', inputnode, 'labels_file')
register.connect(datasource, 'dde_MeanB0_file', inputnode, 'dde_MeanB0_file')


"""
Generate Mean T2-weighted images from all echoes.
Unrotate to reduce rotation/animal effects within the magnet.

"""

t2sagmean = pe.Node(interface=SCTMaths(),name='t2sagmean')
t2sagmean.inputs.mean = 't'
register.connect(inputnode,'t2sagorig_file',t2sagmean,'in_file')

undoXYrotsag = pe.Node(GenUnrotateMatrix(),name='undoXYrotsag')
undoXYrotsag.inputs.randHack = rotval
register.connect(t2sagmean, 'out_file',undoXYrotsag,'in_file')
register.connect(t2sagmean, 'out_file',undoXYrotsag,'rot_file')
register.connect(infosource, ('session_id', getRotInfo),undoXYrotsag,'rot180')

undoXYrotlab = pe.Node(GenUnrotateMatrix(),name='undoXYrotlab')
undoXYrotlab.inputs.randHack = rotval
register.connect(inputnode, 'labels_file',undoXYrotlab,'in_file')
register.connect(t2sagmean, 'out_file',undoXYrotlab,'rot_file')
register.connect(infosource, ('session_id', getRotInfo),undoXYrotlab,'rot180')


# first -pass to just align the labels.  This is fed to the axial registration later, so keep it seperate
# from the full sagittal registration.
t2sag_reglabels = pe.Node(interface=SCTRegisterToTemplate(),name='t2sag_reglabels')
t2sag_reglabels.inputs.param = ('step=0,type=label,dof=Tx_Ty_Tz_Sz'
                             +':step=1,type=im,algo=translation,metric=MI,iter=1,smooth=0.3,gradStep=0.1,slicewise=0,deformation=0x1x0'
                             )
t2sag_reglabels.inputs.centerline_algo = 'polyfit'  #this is a 5th order polynomial
t2sag_reglabels.inputs.centerline_smooth = 20
t2sag_reglabels.inputs.contrast = 't2'
t2sag_reglabels.inputs.template_dir = os.path.abspath(os.path.dirname(__file__)) + '/Templates/RatHistoAtlas'
#the labels are created on the sagittal, so this has to use sagittal, otherwise errors occur since the labels are outside of the moving image/data
register.connect(undoXYrotsag,'out_file',t2sag_reglabels,'in_file')
register.connect(undoXYrotlab,'out_file',t2sag_reglabels,'seg_file')
register.connect(undoXYrotlab,'out_file',t2sag_reglabels,'lspinal_file')



"""
Align the T2 mean image to the t2 template, starting with the labels for initialization
"""

t2sag_regFinal = pe.Node(interface=SCTRegisterMultiModal(),name='t2sag_regFinal')
t2sag_regFinal.inputs.param = ('step=1,type=im,algo=translation,metric=MI,iter=10,shrink=4,smooth=0.3,gradStep=0.3,slicewise=0'
                             +':step=2,type=im,algo=bsplinesyn,metric=MI,iter=5,shrink=4,smooth=0.3,gradStep=0.3,slicewise=0,deformation=0x1x0'
                             )
t2sag_regFinal.inputs.mask_file = templatepath + '/RatHistoAtlas_regMask_cervical.nii.gz'
t2sag_regFinal.inputs.dest_file = templatepath + '/RatHistoAtlas_t2_cervical.nii.gz'
register.connect(t2sag_reglabels,'warp_file', t2sag_regFinal, 'initwarp_file')
register.connect(undoXYrotsag,'out_file',t2sag_regFinal, 'in_file')


undoXYrott2sag = pe.Node(GenUnrotateMatrix(),name='undoXYrott2sag')
undoXYrott2sag.inputs.randHack = rotval
register.connect(inputnode,'t2sag_T2map_file',undoXYrott2sag,'in_file')
register.connect(t2sagmean, 'out_file',undoXYrott2sag,'rot_file')
register.connect(infosource, ('session_id', getRotInfo),undoXYrott2sag,'rot180')

regt2_sag = pe.Node(interface=SCTApplyTransfo(),name='regt2_sag')
register.connect(undoXYrott2sag,'out_file',regt2_sag,'in_file')
regt2_sag.inputs.dest_file = templatepath + '/RatHistoAtlas_t2.nii.gz'
register.connect(t2sag_regFinal,'warp_file',regt2_sag,'warp_file')



"""
Crop images to cervical-only for easier viewing and stats
"""
regt2_sagcrop = pe.Node(interface=SCTCrop(), name='regt2_sagcrop')
regt2_sagcrop.inputs.zmin = "428"
regt2_sagcrop.inputs.zmax = "559"
register.connect(regt2_sag,'out_file',regt2_sagcrop,'in_file')


"""
Axial images, same principle, start with the label warp
"""

t2axmean = pe.Node(interface=SCTMaths(),name='t2axmean')
t2axmean.inputs.mean = 't'
register.connect(inputnode,'t2axorig_file',t2axmean,'in_file')

undoXYrotax = pe.Node(GenUnrotateMatrix(),name='undoXYrotax')
undoXYrotax.inputs.randHack = rotval
register.connect(t2axmean, 'out_file',undoXYrotax,'in_file')
register.connect(t2sagmean, 'out_file',undoXYrotax,'rot_file')
register.connect(infosource, ('session_id', getRotInfo),undoXYrotax,'rot180')


t2ax_reg = pe.Node(interface=SCTRegisterMultiModal(),name='t2ax_reg')
t2ax_reg.inputs.param = ('step=1,type=im,algo=rigid,metric=MI,iter=5,smooth=0.3,shrink=4,gradStep=0.3,slicewise=0'
                         +':step=2,type=im,algo=bsplinesyn,metric=MI,iter=2,smooth=0.3,shrink=8,gradStep=0.3,slicewise=0,deformation=1x1x0'
                            )
t2ax_reg.inputs.dest_file = templatepath + '/RatHistoAtlas_t2_cervical.nii.gz'
t2ax_reg.inputs.mask_file = templatepath + '/RatHistoAtlas_regMask_cervical.nii.gz'
register.connect(undoXYrotax,'out_file',t2ax_reg,'in_file')
register.connect(t2sag_reglabels,'warp_file',t2ax_reg,'initwarp_file')



undoXYrott2ax = pe.Node(GenUnrotateMatrix(),name='undoXYrott2ax')
undoXYrott2ax.inputs.randHack = rotval
register.connect(inputnode,'t2ax_T2map_file',undoXYrott2ax,'in_file')
register.connect(t2sagmean, 'out_file',undoXYrott2ax,'rot_file')
register.connect(infosource, ('session_id', getRotInfo),undoXYrott2ax,'rot180')

regt2_ax = pe.Node(interface=SCTApplyTransfo(),name='regt2_ax')
register.connect(undoXYrott2ax,'out_file',regt2_ax,'in_file')
regt2_ax.inputs.dest_file = templatepath + '/RatHistoAtlas_t2.nii.gz'
register.connect(t2ax_reg,'warp_file',regt2_ax,'warp_file')



"""
Attempt to correct b0/EPI distortions.
Tried registering to the T2 native, T2 template, or dwi to T1 template, but without much success.
The likely best scenario is to use the b0 fieldmap and native space, but this is not integrated or well-tested.
FLS's fugue can do this, but haven't used it much. topup may be a better solution too.
Don't use this for now
"""
#
# b0dwiax_sep = pe.Node(interface=SCTSeparateB0DWI(),name='b0dwiax_sep')
# b0dwiax_sep.inputs.bvecs_file = os.path.join(configpath,'dwiax.bvecs')
# b0dwiax_sep.inputs.bvals_file = os.path.join(configpath,'dwiax.True_NotFitting.bvals')
# register.connect(inputnode,'dwiaxorig_file',b0dwiax_sep,'in_file')
#
#
# b0_reg = pe.Node(interface=SCTRegisterMultiModal(),name='b0_reg')
# b0_reg.inputs.param = ('step=1,type=im,algo=syn,metric=MI,iter=5,smooth=0.5,shrink=4,gradStep=0.6,slicewise=0'
#                         )
# b0_reg.inputs.dest_file = os.path.abspath(os.path.dirname(__file__)) + '/Templates/RatHistoAtlas/template/RatHistoAtlas_t1.nii.gz'
# b0_reg.inputs.mask_file = os.path.abspath(os.path.dirname(__file__)) + '/Templates/RatHistoAtlas/template/RatHistoAtlas_regMask.nii.gz'
# #register.connect(t2axmean,'out_file',b0_reg,'dest_file')
# register.connect(b0dwiax_sep,'dwi_file',b0_reg,'in_file')
# register.connect(t2ax_reg,'warp_file',b0_reg,'initwarp_file')


#mergewarp = pe.Node(interface=util.Merge(2), name="mergewarp")
#register.connect(b0_reg,'warp_file',mergewarp,'in1')
#register.connect(t2ax_reg,'warp_file',mergewarp,'in2')

#axcompositewarp = pe.Node(interface=SCTConcatTransform(), name='axcompositewarp')
#register.connect(mergewarp,'out',axcompositewarp,'in_files')
#register.connect(t2sag_reg,'out_file',axcompositewarp,'dest_file')


"""
As above, use the dwi perpendicular image to register to the t1 template
"""

undoXYrotddeMB0 = pe.Node(GenUnrotateMatrix(),name='undoXYrotddeMB0')
undoXYrotddeMB0.inputs.randHack = rotval
register.connect(inputnode,'dde_MeanB0_file',undoXYrotddeMB0,'in_file')
register.connect(t2sagmean, 'out_file',undoXYrotddeMB0,'rot_file')
register.connect(infosource, ('session_id', getRotInfo),undoXYrotddeMB0,'rot180')

dde_regFinal = pe.Node(interface=SCTRegisterMultiModal(),name='dde_regFinal')
dde_regFinal.inputs.param = ('step=1,type=im,algo=rigid,metric=MeanSquares,iter=10,smooth=0.3,shrink=4,gradStep=0.3,slicewise=0'
                         +':step=2,type=im,algo=bsplinesyn,metric=MeanSquares,iter=2,smooth=0.3,shrink=8,gradStep=0.3,slicewise=0,deformation=1x1x0'
                            )
dde_regFinal.inputs.mask_file = templatepath + '/RatHistoAtlas_regMask_cervical.nii.gz'
dde_regFinal.inputs.dest_file = templatepath + '/RatHistoAtlas_t1_cervical.nii.gz'
register.connect(t2sag_reglabels,'warp_file', dde_regFinal, 'initwarp_file')
register.connect(undoXYrotddeMB0,'out_file',dde_regFinal, 'in_file')

"""
Apply to all of the other axial maps
"""

undoXYrotdde = pe.Node(GenUnrotateMatrix(),name='undoXYrotdde')
undoXYrotdde.inputs.randHack = rotval
register.connect(inputnode,'dde_Daxial_file',undoXYrotdde,'in_file')
register.connect(t2sagmean, 'out_file',undoXYrotdde,'rot_file')
register.connect(infosource, ('session_id', getRotInfo),undoXYrotdde,'rot180')

regdde_ax = pe.Node(interface=SCTApplyTransfo(),name='regdde_ax')
register.connect(undoXYrotdde,'out_file',regdde_ax,'in_file')
regdde_ax.inputs.dest_file = templatepath + '/RatHistoAtlas_t2.nii.gz'
register.connect(dde_regFinal,'warp_file',regdde_ax,'warp_file')


undoXYrotdti = pe.Node(GenUnrotateMatrix(),name='undoXYrotdti')
undoXYrotdti.inputs.randHack = rotval
register.connect(inputnode,'dti_Daxial_file',undoXYrotdti,'in_file')
register.connect(t2sagmean, 'out_file',undoXYrotdti,'rot_file')
register.connect(infosource, ('session_id', getRotInfo),undoXYrotdti,'rot180')

regdti_ax = pe.Node(interface=SCTApplyTransfo(),name='regdti_ax')
register.connect(undoXYrotdti,'out_file',regdti_ax,'in_file')
regdti_ax.inputs.dest_file = templatepath + '/RatHistoAtlas_t2.nii.gz'
register.connect(dde_regFinal,'warp_file',regdti_ax,'warp_file')



undoXYrotdtifa = pe.Node(GenUnrotateMatrix(),name='undoXYrotdtifa')
undoXYrotdtifa.inputs.randHack = rotval
register.connect(inputnode,'dti_fa_file',undoXYrotdtifa,'in_file')
register.connect(t2sagmean, 'out_file',undoXYrotdtifa,'rot_file')
register.connect(infosource, ('session_id', getRotInfo),undoXYrotdtifa,'rot180')

regdtifa_ax = pe.Node(interface=SCTApplyTransfo(),name='regdtifa_ax')
register.connect(undoXYrotdtifa,'out_file',regdtifa_ax,'in_file')
regdtifa_ax.inputs.dest_file = templatepath + '/RatHistoAtlas_t2.nii.gz'
register.connect(dde_regFinal,'warp_file',regdtifa_ax,'warp_file')



undoXYrotdtimd = pe.Node(GenUnrotateMatrix(),name='undoXYrotdtimd')
undoXYrotdtimd.inputs.randHack = rotval
register.connect(inputnode,'dti_md_file',undoXYrotdtimd,'in_file')
register.connect(t2sagmean, 'out_file',undoXYrotdtimd,'rot_file')
register.connect(infosource, ('session_id', getRotInfo),undoXYrotdtimd,'rot180')

regdtimd_ax = pe.Node(interface=SCTApplyTransfo(),name='regdtimd_ax')
register.connect(undoXYrotdtimd,'out_file',regdtimd_ax,'in_file')
regdtimd_ax.inputs.dest_file = templatepath + '/RatHistoAtlas_t2.nii.gz'
register.connect(dde_regFinal,'warp_file',regdtimd_ax,'warp_file')

"""
Crop images to cervical-only for easier viewing and stats
"""

regdde_axcrop = pe.Node(interface=SCTCrop(), name='regdde_axcrop')
#regdwi_axcrop.inputs.mask = os.path.abspath(os.path.dirname(__file__)) + '/Templates/RatHistoAtlas/template/RatHistoAtlas_levels_cervical.nii.gz'
regdde_axcrop.inputs.zmin = "428"
regdde_axcrop.inputs.zmax = "559"
register.connect(regdde_ax,'out_file',regdde_axcrop,'in_file')

regdti_axcrop = pe.Node(interface=SCTCrop(), name='regdti_axcrop')
#regdwi_axcrop.inputs.mask = os.path.abspath(os.path.dirname(__file__)) + '/Templates/RatHistoAtlas/template/RatHistoAtlas_levels_cervical.nii.gz'
regdti_axcrop.inputs.zmin = "428"
regdti_axcrop.inputs.zmax = "559"
register.connect(regdti_ax,'out_file',regdti_axcrop,'in_file')

regdtifa_axcrop = pe.Node(interface=SCTCrop(), name='regdtifa_axcrop')
#regdwi_axcrop.inputs.mask = os.path.abspath(os.path.dirname(__file__)) + '/Templates/RatHistoAtlas/template/RatHistoAtlas_levels_cervical.nii.gz'
regdtifa_axcrop.inputs.zmin = "428"
regdtifa_axcrop.inputs.zmax = "559"
register.connect(regdtifa_ax,'out_file',regdtifa_axcrop,'in_file')

regdtimd_axcrop = pe.Node(interface=SCTCrop(), name='regdtimd_axcrop')
#regdwi_axcrop.inputs.mask = os.path.abspath(os.path.dirname(__file__)) + '/Templates/RatHistoAtlas/template/RatHistoAtlas_levels_cervical.nii.gz'
regdtimd_axcrop.inputs.zmin = "428"
regdtimd_axcrop.inputs.zmax = "559"
register.connect(regdtimd_ax,'out_file',regdtimd_axcrop,'in_file')

regt2_axcrop = pe.Node(interface=SCTCrop(), name='regt2_axcrop')
#regt2_axcrop.inputs.mask = os.path.abspath(os.path.dirname(__file__)) + '/Templates/RatHistoAtlas/template/RatHistoAtlas_levels_cervical.nii.gz'
regt2_axcrop.inputs.zmin = "428"
regt2_axcrop.inputs.zmax = "559"
register.connect(regt2_ax,'out_file',regt2_axcrop,'in_file')


"""
Rename all the outputs for easier file navigation
"""

renamet2Ax = pe.Node(interface=Rename(), name='renamet2Ax')
renamet2Ax.inputs.format_string = 'T2Map_Axial_reg.nii.gz'
register.connect(regt2_axcrop, 'out_file', renamet2Ax, 'in_file')

renamedtiAx = pe.Node(interface=Rename(), name='renamedtiAx')
renamedtiAx.inputs.format_string = 'DTI_Daxial_reg.nii.gz'
register.connect(regdti_axcrop, 'out_file', renamedtiAx, 'in_file')

renamedtifa = pe.Node(interface=Rename(), name='renamedtifa')
renamedtifa.inputs.format_string = 'DTI_FA_reg.nii.gz'
register.connect(regdtifa_axcrop, 'out_file', renamedtifa, 'in_file')

renamedtimd = pe.Node(interface=Rename(), name='renamedtimd')
renamedtimd.inputs.format_string = 'DTI_MD_reg.nii.gz'
register.connect(regdtimd_axcrop, 'out_file', renamedtimd, 'in_file')

renameddeAx = pe.Node(interface=Rename(), name='renameddeAx')
renameddeAx.inputs.format_string = 'DDE_Daxial_reg.nii.gz'
register.connect(regdde_axcrop, 'out_file', renameddeAx, 'in_file')


renamet2Sag = pe.Node(interface=Rename(), name='renamet2Sag')
renamet2Sag.inputs.format_string = 'T2Map_Sag_reg.nii.gz'
register.connect(regt2_sagcrop, 'out_file', renamet2Sag, 'in_file')


"""
End of registration nodes
"""

"""
Connect all processed files to the templatespace output location
"""

datasink = pe.Node(interface=nio.DataSink(), name='datasink')
datasink.inputs.base_directory = os.path.abspath('templatespace')
datasink.inputs.substitutions = [('_subject_id_', '_'),('_session_id_', '')]


register.connect(renamet2Ax,'out_file',datasink,'t2')
# register.connect(t2sag_reg,'warp_file',datasink,'t2.@warp')
register.connect(renamedtiAx,'out_file',datasink,'dti')
register.connect(renamedtifa,'out_file',datasink,'dti.@fa')
register.connect(renamedtimd,'out_file',datasink,'dti.@md')
register.connect(renameddeAx,'out_file',datasink,'dde')
register.connect(dde_regFinal,'warp_file',datasink,'dde.@warp')

register.connect(renamet2Sag,'out_file',datasink,'t2.@sag')

if __name__ == '__main__':


    register.run()
    #register.write_graph()
