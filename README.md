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
  SCTCommands: nipype interface for spinal cord toolbox\
  DWIMapsMatlab: nipype interface to DWI matlab processing\
  T2MapMatlab: nipype interface to T2 maps in matlab\

  Matlab processing files:\
    T2map.m: matlab code to estimate T2\
    DWI_fADCmap.m: spinal cord specific DWI (ADC parallel) map estimation\

# Other Notes:
  Nipype technical details: Nipype interfaces were created for the spinal cord toolbox for this project. I only included the functions and features I needed in this pipeline, but it is a start and quite useful in my opinion. It should be grown out to a full-fledged nipype interface and included in the nipype package, in my option, but would also need to be maintained as new options are added to SCT. 
  It uses qMRIlab for T1 mapping and is implemented as a submodule in github.  May need to run `git submodule update --remote` from within the MatlabTools/qMRIlab folder to download the newest version.
