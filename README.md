# RatSpinalCordMRIpipeline
Processing template for rat spinal cord MRI in nipype
PI: Matthew Budde


These routines process animal MRI data from nifti raw images to parameter maps registered to a Rat-specific template

There is 1 manual step:
  Placing spinal cord labels: these are necessary for the registration step and was more robust compared to segmentation approaches.

## Setup
  It has been tested on MacOS.  It requires installation of Matlab, ants, spinalcordtoolbox, and FSL.\


## In order, here are the steps:
  * Create a Master processing file (csv format: [MRIscanlist.csv](https://github.com/mdbudde/BuddeDoDMRIprocessing/MRIscanlist.csv)) in the format of:\
        Subject,Session,t2sag,t2ax,mge,dti,dde,rot180 \
        CSCI07,CSCI07_1D,3,6,9,10,24,True \
        CSCI08,CSCI08_1D,3,4,5,7,6,True \
        CSCI09,CSCI09_1D,3,4,5,7,6,True \
        ...\
    where the subject/session is identical to that input on the scanner, and the numbers indicate the scan number (bruker) for each metric/scan. The scan number helps to find and convert from the MRI protocol data to a more generic notation (BIDS-like, but not quite).   The last column specifies whether to  rotate the volumes by 180 degrees or not.  On the Bruker scanner, the user selects Supine/Prone, and the images will be rotated with respect to the correct orientation if not properly selected.


  * Process all maps, DWI, T2, MGE. This uses matlab and qMRIlab and FSL, protocol-specific inputs to these tools are in this file for each scan type. In the scanlist, an additional column can be added with a True or 1 flag to indicate the entire set of images should be rotated by 180 degrees if the wrong prone/supine was used on the scanner.\  
    `python3 Step1_AllPreprocessing.py -f MRIscanlist.csv`



  * Create manual labels in the spinal cord for registration. When the viewer opens, place the labels in the center of the cord (AP) and at the CENTER BETWEEN TWO DISKS (SI). Note a -r flag can be added to update the labels if they need adjustment/additions.
    `python3 Step2_CreateVertLabels.py -f MRIscanlist.csv`



  * Register all converted maps to template using labels to initialize\
    `python3 Step3_AllRegistration.py -f MRIscanlist.csv`

  * All of the processed/registered images will be stored in the template/metric/session_subject/ folders and can be used for subsequent stats.

  * A set of pipelines also setup to configure for statistical test (FSL randomise) with many different contrasts and configurations.  Currently this is Setup to do regression analysis with covariates of interest, but could be extended for other purposes.\
    `python3 Step4_AllRegistration.py -f MRIscanlist.csv`


## Other secondary files:
  SCTCommands: nipype interface for spinal cord toolbox. Note that we have included many (but not all) of the sct functions/scripts into a nipype interface. It would be really helpful to have these available by the authors of the code (hint, hint) given the widespread use of nipype in the MRI community.\
  DWIMapsMatlab: nipype interface to DWI matlab processing\
  T2MapMatlab: nipype interface to T2 maps in matlab\

  Matlab processing files:\
    T2map.m: matlab code to estimate T2\
    DWI_fADCmap.m: spinal cord specific DWI (ADC parallel) map estimation\

# Other Notes:
  Nipype technical details: Nipype interfaces were created for the spinal cord toolbox for this project. I only included the functions and features I needed in this pipeline, but it is a start and quite useful in my opinion. It should be grown out to a full-fledged nipype interface and included in the nipype package, in my option, but would also need to be maintained as new options are added to SCT. 
  It uses qMRIlab for T1 mapping and is implemented as a submodule in github.  May need to run `git submodule update --remote` from within the MatlabTools/qMRIlab folder to download the newest version.


## Spinal Cord Toolbox for rat data:
Two manual updates have been made to the spinal cord toolbox to enable it to function better for rat spinal cords: 
In the file: ${scthome}/spinalcordtoolbox/scripts/sct_register_to_template.py - About line 440: 
change resolution from "1.0x1.0x1.0" to "0.1x0.1x0.1" on the following two lines (note the suffexes can also be changed for accuracy):
- resample_file(ftmp_data, add_suffix(ftmp_data, '_0p1mm'), '0.1x0.1x0.1', 'mm', 'linear', verbose)
- resample_file(ftmp_seg, add_suffix(ftmp_seg, '_0p1mm'), '0.1x0.1x0.1', 'mm', 'linear', verbose)

  
In the file: ${scthome}/spinalcordtoolbox/registration/core.py - About line 290: 
change the default resolution of the registration params (note we use the bsplinesyn option):
- #Modified code: (scale factors by one-tenth)
ants_registration_params = {'rigid': '', 'affine': '', 'compositeaffine': '', 'similarity': '', 'translation': '', 'bspline': ',1', 'gaussiandisplacementfield': ',3,0', 'bsplinedisplacementfield': ',0.2,1', 'syn': ',0.3,0', 'bsplinesyn': ',0.1,0.3'} # MDB, original 
- #Original code:
ants_registration_params = {'rigid': '', 'affine': '', 'compositeaffine': '', 'similarity': '', 'translation': '', # 'bspline': ',10', 'gaussiandisplacementfield': ',3,0', # 'bsplinedisplacementfield': ',5,10', 'syn': ',3,0', 'bsplinesyn': ',1,3'}
