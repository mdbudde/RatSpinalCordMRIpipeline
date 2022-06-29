# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Spinal Cord Toolbox v4.3 commands through nipype

Note that only the commands that we needed for this project are included.
So not all of the sct features and flags are included. All of the sct help output
is included prior to the nipype interface, so it should be easy to add and build on.


"""
import os
import os.path as op
from warnings import warn
from pathlib import Path

import numpy as np
from nibabel import load
from nipype.utils.filemanip import fname_presuffix

from nipype.interfaces.base import (
    CommandLine,
    CommandLineInputSpec,
    TraitedSpec,
    File,
    Directory,
    InputMultiPath,
    traits,
    isdefined
)

"""
--
Spinal Cord Toolbox (4.3)


DESCRIPTION
Register an anatomical image to the spinal cord MRI template (default: PAM50).

  The registration process includes three main registration steps:
  1. straightening of the image using the spinal cord segmentation (see sct_straighten_spinalcord
    for details);
  2. vertebral alignment between the image and the template, using labels along the spine;
  3. iterative slice-wise non-linear registration (see sct_register_multimodal for details)

  To register a subject to the template, try the default command:
  sct_register_to_template -i data.nii.gz -s data_seg.nii.gz -l data_labels.nii.gz

  If this default command does not produce satisfactory results, please refer to:
  https://sourceforge.net/p/spinalcordtoolbox/wiki/registration_tricks/

  The default registration method brings the subject image to the template, which can be
    problematic with highly non-isotropic images as it would induce large interpolation errors dur
    ing the straightening procedure. Although the default method is recommended, you may want to r
    egister the template to the subject (instead of the subject to the template) by skipping the st
    raightening procedure. To do so, use the parameter "-ref subject". Example below:
  sct_register_to_template -i data.nii.gz -s data_seg.nii.gz -l data_labels.nii.gz -ref subject
    -pa
    ram step=1,type=seg,algo=centermassrot,smooth=0:step=2,type=seg,algo=columnwise,smooth=0,smoothWarpXY=2

  Vertebral alignment (step 2) consists in aligning the vertebrae between the subject and the
    template. Two types of labels are possible:
  - Vertebrae mid-body labels, created at the center of the spinal cord using the parameter "-l";
  - Posterior edge of the intervertebral discs, using the parameter "-ldisc".

  If only one label is provided, a simple translation will be applied between the subject label and
    the template label. No scaling will be performed.

  If two labels are provided, a linear transformation (translation + rotation + superior-inferior
    linear scaling) will be applied. The strategy here is to defined labels that cover the region
     of interest. For example, if you are interested in studying C2 to C6 levels, then provide one l
    abel at C2 and another at C6. However, note that if the two labels are very far apart (e.g. C
    2 and T12), there might be a mis-alignment of discs because a subjects intervertebral discs
     distance might differ from that of the template.

  If more than two labels (only with the parameter "-disc") are used, a non-linear registration
    will be applied to align the each intervertebral disc between the subject and the template,
     as described in sct_straighten_spinalcord. This the most accurate and preferred method. This
    feature does not work with the parameter "-ref subject".

  More information about label creation can be found at
    https://www.slideshare.net/neuropoly/sct-course-20190121/42

USAGE
sct_register_to_template -i <file> -s <file>

MANDATORY ARGUMENTS
 -i <file>                    Anatomical image.
 -s <file>                    Spinal cord segmentation.

OPTIONAL ARGUMENTS
 -l <file>                    One or two labels (preferred) located at the center of the spinal
                              cord, on the mid-vertebral slice. For more information about label
                              creation, please see:
                              https://www.slideshare.net/neuropoly/sct-course-20190121/42 Default
                              value =
 -ldisc <file>                Labels located at the posterior edge of the intervertebral discs. If
                              you are using more than 2 labels, all disc covering the region of
                              interest should be provided. E.g., if you are interested in levels C2
                              to C7, then you should provide disc labels 2,3,4,5,6,7). For more
                              information about label creation, please refer to
                              https://www.slideshare.net/neuropoly/sct-course-20190121/42 Default
                              value =
 -lspinal <file>              Labels located in the center of the spinal cord, at the
                              superior-inferior level corresponding to the mid-point of the spinal
                              level. Each label is a single voxel, which value corresponds to the
                              spinal level (e.g.: 2 for spinal level 2). If you are using more than
                              2 labels, all spinal levels covering the region of interest should be
                              provided (e.g., if you are interested in levels C2 to C7, then you
                              should provide spinal level labels 2,3,4,5,6,7). Default value =
 -ofolder <folder_creation>   Output folder. Default value =
 -t <folder>                  Path to template. Default value = /usr/local/sct_v4.3/data/PAM50
 -c {t1,t2,t2s}               Contrast to use for registration. Default value = t2
 -ref {template,subject}      Reference for registration: template: subject->template, subject:
                              template->subject. Default value = template
 -param <list of: str>        Parameters for registration (see sct_register_multimodal). Default:

                                --
                                step=0
                                type=label
                                dof=Tx_Ty_Tz_Sz
                                --
                                step=1
                                type=imseg
                                algo=centermassrot
                                metric=MeanSquares
                                iter=10
                                smooth=0
                                gradStep=0.5
                                slicewise=0
                                smoothWarpXY=2
                                pca_eigenratio_th=1.6
                                --
                                step=2
                                type=seg
                                algo=bsplinesyn
                                metric=MeanSquares
                                iter=3
                                smooth=1
                                gradStep=0.5
                                slicewise=0
                                smoothWarpXY=2
                                pca_eigenratio_th=1.6
 -centerline-algo {polyfit,bspline,linear,nurbs}Algorithm for centerline fitting (when straightening the spinal
                              cord). Default value = bspline
 -centerline-smooth <int>     Degree of smoothing for centerline fitting. Only use with
                              -centerline-algo {bspline, linear}. Default value = 20
 -qc <folder_creation>        The path where the quality control generated content will be saved
 -qc-dataset <str>            If provided, this string will be mentioned in the QC report as the
                              dataset the process was run on
 -qc-subject <str>            If provided, this string will be mentioned in the QC report as the
                              subject the process was run on
 -igt <image_nifti>           File name of ground-truth template cord segmentation (binary nifti).
 -r {0,1}                     Remove temporary files. Default value = 1
 -v {0,1,2}                   Verbose. 0: nothing. 1: basic. 2: extended. Default value = 1

"""


class SCTRegisterToTemplate_InputSpec(CommandLineInputSpec):
    in_file = File(exists=True, argstr="-i \"%s\"",
                   mandatory=True, position=0, desc="input file")
    lspinal_file = File(exists=True, argstr="-lspinal \"%s\"",
                        mandatory=False, desc="lspinal file")
    seg_file = File(exists=True, argstr="-s \"%s\"",
                    mandatory=False, position=1, desc="seg file")
    label_file = File(exists=True, argstr="-l \"%s\"",
                      mandatory=False, desc="label file")
    template_dir = Directory(exists=True, argstr="-t \"%s\"",
                             resolve=True, mandatory=False, desc="template dir")
    centerline_algo = traits.Str(argstr="-centerline-algo %s",
                                 mandatory=False, desc="centerline_algo", default='bspline')
    centerline_smooth = traits.Int(
        argstr="-centerline-smooth %s", mandatory=False, desc="centerline_smooth", default=20)
    contrast = traits.Str(argstr="-c %s", mandatory=False,
                          desc="contrast", default='t2')
    param = traits.Str(argstr="-param %s",
                       desc="string defining the paramters, i. e. -param ...")


class SCTRegisterToTemplate_OutputSpec(TraitedSpec):
    out_file = File(exists=False)
    warp_file = File(exists=False)


class SCTRegisterToTemplate(CommandLine):
    _cmd = "sct_register_to_template "
    input_spec = SCTRegisterToTemplate_InputSpec
    output_spec = SCTRegisterToTemplate_OutputSpec

    def _run_interface(self, runtime, correct_return_codes=(0,)):
        runtime = super(SCTRegisterToTemplate, self)._run_interface(runtime)
        return runtime

    def aggregate_outputs(self, runtime=None, needed_outputs=None):
        outputs = super(SCTRegisterToTemplate, self).aggregate_outputs(
            runtime=runtime, needed_outputs=needed_outputs
        )
        return outputs

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['warp_file'] = os.path.abspath("warp_anat2template.nii.gz")
        origsuffix = Path(self.inputs.in_file).suffixes
        outputs['out_file'] = os.path.abspath(
            "anat2template" + ''.join(origsuffix))
        return outputs


"""
--
Spinal Cord Toolbox (4.3)


DESCRIPTION
This program co-registers two 3D volumes. The deformation is non-rigid and is constrained along Z
direction (i.e., axial plane). Hence, this function assumes that orientation of the destination
image is axial (RPI). If you need to register two volumes with large deformations and/or different
contrasts, it is recommended to input spinal cord segmentations (binary mask) in order to achieve
maximum robustness. The program outputs a warping field that can be used to register other images
to the destination image. To apply the warping field to another image, use sct_apply_transfo

USAGE
sct_register_multimodal -i <file> -d <file>

MANDATORY ARGUMENTS
 -i <file>                    Image source.
 -d <file>                    Image destination.

OPTIONAL ARGUMENTS
 -iseg <file>                 Segmentation source.
 -dseg <file>                 Segmentation destination.
 -ilabel <file>               Labels source.
 -dlabel <file>               Labels destination.
 -initwarp <file>             Initial warping field to apply to the source image.
 -initwarpinv <file>          Initial inverse warping field to apply to the destination image (only
                              use if you wish to generate the dest->src warping field).
 -m <file>                    Mask that can be created with sct_create_mask to improve accuracy
                              over region of interest. This mask will be used on the destination
                              image.
 -o <file_output>             Name of output file.
 -owarp <file_output>         Name of output forward warping field.
 -param <list of: str>        Parameters for registration. Separate arguments with ",". Separate
                              steps with ":".
                                step: <int> Step number (starts at 1, except for type=label).
                                type: {im, seg, imseg, label} type of data used for registration.
                                  Use type=label only at step=0.
                                algo: Default=syn
                                  translation: translation in X-Y plane (2dof)
                                  rigid: translation + rotation in X-Y plane (4dof)
                                  affine: translation + rotation + scaling in X-Y plane (6dof)
                                  syn: non-linear symmetric normalization
                                  bsplinesyn: syn regularized with b-splines
                                  slicereg: regularized translations (see: goo.gl/Sj3ZeU)
                                  centermass: slicewise center of mass alignment (seg only).
                                  centermassrot: slicewise center of mass and rotation alignment
                                  using method specified in 'rot_method'
                                  columnwise: R-L scaling followed by A-P columnwise alignment (seg
                                  only).
                                slicewise: <int> Slice-by-slice 2d transformation. Default=0
                                metric: {
                                    CC,MI,MeanSquares}. Default=MeanSquares
                                iter: <int> Number of iterations. Default=10
                                shrink: <int> Shrink factor (only for syn/bsplinesyn). Default=1
                                smooth: <int> Smooth factor (in mm). Note: if
                                  algo={centermassrot,columnwise} the smoothing kernel is: SxS
                                  x0. Otherwise it is SxSxS. Default=0
                                laplacian: <int> Laplacian filter. Default=0
                                gradStep: <float> Gradient step. Default=0.5
                                deformation: ?x?x?: Restrict deformation (for ANTs algo). Replace ?
                                  by 0 (no deformation) or 1 (deformation). Default=1x1x0
                                init: Initial translation alignment based on:
                                  geometric: Geometric center of images
                                  centermass: Center of mass of images
                                  origin: Physical origin of images
                                poly: <int> Polynomial degree of regularization (only for
                                  algo=slicereg). Default=5
                                filter_size: <float> Filter size for regularization (only for
                                  algo=centermassrot). Default=5
                                smoothWarpXY: <int> Smooth XY warping field (only for
                                  algo=columnwize). Default=2
                                pca_eigenratio_th: <int> Min ratio between the two eigenvalues for
                                  PCA-based angular adjustment (only for algo=centermassrot
                                  and rot_method=pca). Default=1.6
                                dof: <str> Degree of freedom for type=label. Separate with '_'.
                                  Default=Tx_Ty_Tz_Rx_Ry_Rz
                                pca
                                rot_method {pca, hog, pcahog}: rotation method to be used with
                                  algo=centermassrot. pca: approximate cord segmentation by an elli
                                  pse and finds it orientation using PCA's eigenvectors; hog: fi
                                  nds the orientation using the symmetry of the image; pcahog: tries
                                   method pca and if it fails, uses method hog. If using hog or pcah
                                  og, type should be set to imseg.
 -identity {0,1}              just put source into destination (no optimization). Default value = 0
 -z <int>                     size of z-padding to enable deformation at edges when using SyN.
                              Default value = 5
 -x {nn,linear,spline}        Final interpolation. Default value = linear
 -ofolder <folder_creation>   Output folder
 -qc <folder_creation>        The path where the quality control generated content will be saved
 -qc-dataset <str>            If provided, this string will be mentioned in the QC report as the
                              dataset the process was run on
 -qc-subject <str>            If provided, this string will be mentioned in the QC report as the
                              subject the process was run on
 -r {0,1}                     Remove temporary files. Default value = 1
 -v {0,1,2}                   Verbose. Default value = 1
"""


class SCTRegisterMultiModal_InputSpec(CommandLineInputSpec):
    in_file = File(exists=True, argstr="-i \"%s\"",
                   mandatory=True, position=0, desc="input file")
    dest_file = File(exists=True, argstr="-d \"%s\"",
                     mandatory=True, desc="destination file")
    iseg_file = File(exists=True, argstr="-iseg \"%s\"",
                     mandatory=False, desc="in seg file")
    dseg_file = File(exists=True, argstr="-dseg \"%s\"",
                     mandatory=False, desc="destination seg file")
    ilabel_file = File(exists=True, argstr="-ilabel \"%s\"",
                       mandatory=False, desc="in label file")
    initwarp_file = File(exists=True, argstr="-initwarp \"%s\"",
                         mandatory=False, desc="init warpfile")
    dlabel_file = File(exists=True, argstr="-dlabel \"%s\"",
                       mandatory=False, desc="destination label file")
    mask_file = File(exists=True, argstr="-m \"%s\"",
                     mandatory=False, desc="mask file")
    out_file = File(exists=True, argstr="-o \"%s\"",
                    mandatory=False, desc="out file")
    zpad = traits.Str(exists=True, argstr="-z \"%s\"",
                      mandatory=False, desc="z padding for SyN")
    param = traits.Str(argstr="-param %s",
                       desc="string defining the paramters, i. e. -param ...")


class SCTRegisterMultiModal_OutputSpec(TraitedSpec):
    out_file = File(exists=False)
    warp_file = File(exists=False)


class SCTRegisterMultiModal(CommandLine):
    _cmd = "sct_register_multimodal -owarp warp_anat2multimodal.nii.gz "
    input_spec = SCTRegisterMultiModal_InputSpec
    output_spec = SCTRegisterMultiModal_OutputSpec

    def _run_interface(self, runtime, correct_return_codes=(0,)):
        runtime = super(SCTRegisterMultiModal, self)._run_interface(runtime)
        return runtime

    def aggregate_outputs(self, runtime=None, needed_outputs=None):
        outputs = super(SCTRegisterMultiModal, self).aggregate_outputs(
            runtime=runtime, needed_outputs=needed_outputs
        )
        return outputs

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['warp_file'] = os.path.abspath("warp_anat2multimodal.nii.gz")
        outputs['out_file'] = os.path.abspath(
            self.inputs.in_file.replace(".nii", "_reg.nii"))
        return outputs


"""
--
Spinal Cord Toolbox (4.3)


DESCRIPTION
This function extracts the spinal cord centerline. Three methods are
      available: OptiC (automatic), Viewer (manual) and Fitseg (applied on segmented image). These
    functions output (i) a NIFTI file with labels corresponding
      to the discrete centerline, and (ii) a csv file containing the float (more precise)
    coordinates of the centerline
      in the RPI orientation.

  Reference: C Gros, B De Leener, et al. Automatic spinal cord
      localization, robust to MRI contrast using global curve optimization (2017).
    doi.org/10.1016/j.media.2017.12.001

USAGE
sct_get_centerline -i <image_nifti>

MANDATORY ARGUMENTS
 -i <image_nifti>             Input image.

OPTIONAL ARGUMENTS
 -c {t1,t2,t2s,dwi}           Type of image contrast. Only with method=optic.
 -method {optic,viewer,fitseg}Method used for extracting the centerline.
                                optic: automatic spinal cord detection method
                                viewer: manual selection a few points followed by interpolation
                                fitseg: fit a regularized centerline on an already-existing cord
                                  segmentation. It will interpolate if slices are missing
                                  and extrapolate beyond the segmentation boundaries (i.e., ever
                                  y axial slice will exhibit a centerline pixel). Default value = optic
 -centerline-algo {polyfit,bspline,linear,nurbs}Algorithm for centerline fitting. Only relevant with -angle-corr 1.
                              Default value = bspline
 -centerline-smooth <int>     Degree of smoothing for centerline fitting. Only for -centerline-algo
                              {bspline, linear}. Default value = 30
 -o <file_output>             File name (without extension) for the centerline output files. By
                              default, outputfile will be the input with suffix "_centerline"
 -gap <float>                 Gap in mm between manually selected points. Only with method=viewer.
                              Default value = 20.0
 -igt <image_nifti>           File name of ground-truth centerline or segmentation (binary nifti).
 -v {0,1,2}                   1: display on, 0: display off (default) Default value = 1
"""


class SCTGetCenterline_InputSpec(CommandLineInputSpec):
    in_file = File(exists=True, argstr="-i \"%s\"",
                   mandatory=True, position=0, desc="input file")
    method = traits.Str(argstr="-method %s",
                        desc="method {optic,viewer,fitseg}")


class SCTGetCenterline_OutputSpec(TraitedSpec):
    out_file = File(exists=False)


class SCTGetCenterline(CommandLine):
    _cmd = "sct_get_centerline "
    input_spec = SCTGetCenterline_InputSpec
    output_spec = SCTGetCenterline_OutputSpec

    def _run_interface(self, runtime, correct_return_codes=(0,)):
        runtime = super(SCTGetCenterline, self)._run_interface(runtime)
        return runtime

    def aggregate_outputs(self, runtime=None, needed_outputs=None):
        outputs = super(SCTGetCenterline, self).aggregate_outputs(
            runtime=runtime, needed_outputs=needed_outputs
        )
        return outputs

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = os.path.abspath(
            self.inputs.in_file.replace(".nii", "_centerline.nii"))
        return outputs


"""
--
Spinal Cord Toolbox (4.3)

usage: sct_straighten_spinalcord -i <file> -s <file> [-h] [-dest <file>] [-ldisc-input <file>] [-ldisc-dest <file>] [-disable-straight2curved]
                                 [-disable-curved2straight] [-speed-factor <float>] [-xy-size <float>] [-o <file>] [-ofolder <folder>]
                                 [-centerline-algo {bspline,linear,nurbs}] [-centerline-smooth <int>] [-param <list>] [-x {nn,linear,spline}] [-qc <str>]
                                 [-qc-dataset <str>] [-qc-subject <str>] [-r {0,1}] [-v {0,1,2}]

This program takes as input an anatomic image and the spinal cord centerline (or segmentation), and returns the an image of a straightened spinal cord. Reference: De Leener B, Mangeat G, Dupont S, Martin AR, Callot V, Stikov N, Fehlings MG, Cohen-Adad J. Topologically-preserving straightening of spinal cord MRI. J Magn Reson Imaging. 2017 Oct;46(4):1209-1219

MANDATORY ARGUMENTS:
  -i <file>             Input image with curved spinal cord. Example: "t2.nii.gz"
  -s <file>             Spinal cord centerline (or segmentation) of the input image. To obtain the centerline, you can use sct_get_centerline. To obtain the
                        segmentation you can use sct_propseg or sct_deepseg_sc. Example: centerline.nii.gz

OPTIONAL ARGUMENTS:
  -h, --help            Show this help message and exit
  -dest <file>          Spinal cord centerline (or segmentation) of a destination image (which could be straight or curved). An algorithm scales the length
                        of the input centerline to match that of the destination centerline. If using -ldisc_input and -ldisc_dest with this parameter,
                        instead of linear scaling, the source centerline will be non-linearly matched so that the inter-vertebral discs of the input image
                        will match that of the destination image. This feature is particularly useful for registering to a template while accounting for
                        disc alignment.
  -ldisc-input <file>   Labels located at the posterior edge of the intervertebral discs, for the input image (-i). All disc covering the region of interest
                        should be provided. Exmaple: if you are interested in levels C2 to C7, then you should provide disc labels 2,3,4,5,6,7). More
                        details about label creation at http://sourceforge.net/p/spinalcordtoolbox/wiki/create_labels/. This option must be used with the
                        -ldisc_dest parameter.
  -ldisc-dest <file>    Labels located at the posterior edge of the intervertebral discs, for the destination file (-dest). The same comments as in
                        -ldisc_input apply. This option must be used with the -ldisc_input parameter.
  -disable-straight2curved
                        Disable straight to curved transformation computation, in case you do not need the output warping field straight-->curve (faster).
  -disable-curved2straight
                        Disable curved to straight transformation computation, in case you do not need the output warping field curve-->straight (faster).
  -speed-factor <float>
                        Acceleration factor for the calculation of the straightening warping field. This speed factor enables an intermediate resampling to
                        a lower resolution, which decreases the computational time at the cost of lower accuracy. A speed factor of 2 means that the input
                        image will be downsampled by a factor 2 before calculating the straightening warping field. For example, a 1x1x1 mm^3 image will be
                        downsampled to 2x2x2 mm3, providing a speed factor of approximately 8. Note that accelerating the straightening process reduces the
                        precision of the algorithm, and induces undesirable edges effects. Default=1 (no downsampling).
  -xy-size <float>      Size of the output FOV in the RL/AP plane, in mm. The resolution of the destination image is the same as that of the source image
                        (-i).
  -o <file>             Straightened file. By default, the suffix "_straight" will be added to the input file name.
  -ofolder <folder>     Output folder (all outputs will go there).
  -centerline-algo {bspline,linear,nurbs}
                        Algorithm for centerline fitting.
  -centerline-smooth <int>
                        Degree of smoothing for centerline fitting. Only use with -centerline-algo {bspline, linear}.
  -param <list>         Parameters for spinal cord straightening. Separate arguments with ','.
                        precision: [1.0,inf[. Precision factor of straightening, related to the number of slices. Increasing this parameter increases the
                        precision along with increased computational time. Not taken into account with hanning fitting method. Default=2
                        threshold_distance: [0.0,inf[. Threshold at which voxels are not considered into displacement. Increase this threshold if the image
                        is blackout around the spinal cord too much. Default=10
                        accuracy_results: {0, 1} Disable/Enable computation of accuracy results after straightening. Default=0
                        template_orientation: {0, 1} Disable/Enable orientation of the straight image to be the same as the template. Default=0
  -x {nn,linear,spline}
                        Final interpolation.
  -qc <str>             The path where the quality control generated content will be saved
  -qc-dataset <str>     If provided, this string will be mentioned in the QC report as the dataset the process was run on
  -qc-subject <str>     If provided, this string will be mentioned in the QC report as the subject the process was run on
  -r {0,1}              Remove temporary files.
  -v {0,1,2}            Verbose. 0: nothing, 1: basic, 2: extended.
"""


class SCTStraightenSpinalcord_InputSpec(CommandLineInputSpec):
    in_file = File(exists=True, argstr="-i \"%s\"",
                   mandatory=True, position=0, desc="input file")
    center_file = File(exists=True, argstr="-s \"%s\"",
                       mandatory=True, desc="centerline file")
    param = traits.Str(argstr="-param %s",
                       desc="string defining the paramters, i. e. -param ...")


class SCTStraightenSpinalcord_OutputSpec(TraitedSpec):
    out_file = File(exists=False)


class SCTStraightenSpinalcord(CommandLine):
    _cmd = "sct_straighten_spinalcord "
    input_spec = SCTStraightenSpinalcord_InputSpec
    output_spec = SCTStraightenSpinalcord_OutputSpec

    def _run_interface(self, runtime, correct_return_codes=(0,)):
        runtime = super(SCTStraightenSpinalcord, self)._run_interface(runtime)
        return runtime

    def aggregate_outputs(self, runtime=None, needed_outputs=None):
        outputs = super(SCTStraightenSpinalcord, self).aggregate_outputs(
            runtime=runtime, needed_outputs=needed_outputs
        )
        return outputs

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = os.path.abspath(
            self.inputs.in_file.replace(".nii", "_straighten.nii"))
        return outputs


"""
--
Spinal Cord Toolbox (4.3)

usage: sct_apply_transfo -i <file> -d <file> -w <file> [<file> ...] [-winv <file> [<file> ...]] [-h] [-crop {0,1,2}] [-o <file>]
                         [-x {nn,linear,spline,label}] [-r {0,1}] [-v {0,1,2}]

Apply transformations. This function is a wrapper for antsApplyTransforms (ANTs).

MANDATORY ARGUMENTS:
  -i <file>             Input image. Example: t2.nii.gz
  -d <file>             Destination image. Example: out.nii.gz
  -w <file> [<file> ...]
                        Transformation(s), which can be warping fields (nifti image) or affine transformation matrix (text file). Separate with space.
                        Example: warp1.nii.gz warp2.nii.gz

OPTIONAL ARGUMENTS:
  -winv <file> [<file> ...]
                        Affine transformation(s) listed in flag -w which should be inverted before being used. Note that this only concerns affine
                        transformation (not warping fields). If you would like to use an inverse warping field, then directly input the inverse warping
                        field in flag -w.
  -h, --help            Show this help message and exit
  -crop {0,1,2}         Crop Reference. 0: no reference, 1: sets background to 0, 2: use normal background.
  -o <file>             Registered source. Example: dest.nii.gz
  -x {nn,linear,spline,label}
                        Interpolation method. The 'label' method is to be used if you would like to apply a transformation on a file that has single-voxel
                        labels (classical interpolation methods won't work, as resampled labels might disappear or their values be altered). The function
                        will dilate each label, apply the transformation using nearest neighbour interpolation, and then take the center-of-mass of each
                        "blob" and output a single voxel per blob.
  -r {0,1}              Remove temporary files.
  -v {0,1,2}            Verbose: 0: nothing, 1: classic, 2: expended.
"""


class SCTApplyTransfo_InputSpec(CommandLineInputSpec):
    in_file = File(exists=True, argstr="-i \"%s\"",
                   mandatory=True, position=0, desc="input file")
    dest_file = File(exists=True, argstr="-d \"%s\"",
                     mandatory=True, position=1, desc="destination file")
    warp_file = File(exists=True, argstr="-w \"%s\"",
                     mandatory=True, desc="warp files")
    interp = traits.Str(argstr="-x \"%s\"",
                     mandatory=False, desc="interp")
    invwarp_files = traits.List(
        File(exists=True, argstr="-winv \"%s\"", mandatory=False, desc="warp files"))
    out_file = File(genfile=True, argstr="-o \"%s\"",
                    desc="image to write", hash_files=False)


class SCTApplyTransfo_OutputSpec(TraitedSpec):
    out_file = File(exists=False)


class SCTApplyTransfo(CommandLine):
    _cmd = "sct_apply_transfo "
    input_spec = SCTApplyTransfo_InputSpec
    output_spec = SCTApplyTransfo_OutputSpec
    _suffix = "_reg"

    def _run_interface(self, runtime, correct_return_codes=(0,)):
        runtime = super(SCTApplyTransfo, self)._run_interface(runtime)
        return runtime

    def aggregate_outputs(self, runtime=None, needed_outputs=None):
        outputs = super(SCTApplyTransfo, self).aggregate_outputs(
            runtime=runtime, needed_outputs=needed_outputs
     	)
        return outputs

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if not isdefined(self.inputs.out_file):
            outputs["out_file"] = self._gen_fname(
                self.inputs.in_file, suffix=self._suffix
            )
        print("Output file: "+outputs["out_file"])
        outputs["out_file"] = os.path.abspath(outputs["out_file"])
        return outputs

    def _gen_filename(self, name):
        if name == "out_file":
            return self._list_outputs()["out_file"]
        return None

    def _gen_fname(self, basename, cwd=None, suffix=None, change_ext=True, ext=None):
        """Generate a filename based on the given parameters.
        The filename will take the form: cwd/basename<suffix><ext>.
        If change_ext is True, it will use the extentions specified in
        <instance>intputs.output_type.
        Parameters
        ----------
        basename : str
            Filename to base the new filename on.
        cwd : str
            Path to prefix to the new filename. (default is os.getcwd())
        suffix : str
            Suffix to add to the `basename`.  (defaults is '' )
        Returns
        -------
        fname : str
            New filename based on given parameters.
        """

        if basename == "":
            msg = "Unable to generate filename for command %s. " % self.cmd
            msg += "basename is not set!"
            raise ValueError(msg)
        if cwd is None:
            cwd = os.getcwd()
        if ext is None:
            ext = ".nii.gz"
        if change_ext:
            if suffix:
                suffix = "".join((suffix, ext))
            else:
                suffix = ext
        if suffix is None:
            suffix = ""
        fname = fname_presuffix(
            basename, suffix=suffix, use_ext=False, newpath=cwd)
        return fname


"""
--
Spinal Cord Toolbox (4.3)


DESCRIPTION
Utility function for label image.

USAGE
sct_label_utils -i <file>

MANDATORY ARGUMENTS
 -i <file>                    Input image.

OPTIONAL ARGUMENTS
 -add <int>                   Add value to all labels. Value can be negative.
 -create <list of: Coordinate>Create labels in a new image. List labels as:
                              x1,y1,z1,value1:x2,y2,z2,value2, ...
 -create-add <list of: Coordinate>Same as "-create", but add labels to the input image instead of
                              creating a new image.
 -create-seg <list of: str>   Create labels along cord segmentation (or centerline) defined by
                              "-i". First value is "z", second is the value of the label. Separate
                              labels with ":". Example: 5,1:14,2:23,3. To select the mid-point in
                              the superior-inferior direction, set z to "-1". For example if you
                              know that C2-C3 disc is centered in the S-I direction, then enter:
                              -1,3
 -create-viewer <list of: int>Manually label from a GUI a list of labels IDs, separated with ",".
                              Example: 2,3,4,5
 -ilabel <file>               File that contain labels that you want to correct. It is possible to
                              add new points with this option. Use with -create-viewer
 -cubic-to-point              Compute the center-of-mass for each label value.
 -display                     Display all labels (i.e. non-zero values).
 -increment                   Takes all non-zero values, sort them along the inverse z direction,
                              and attributes the values 1, 2, 3, etc.
 -vert-body <list of: int>    From vertebral labeling, create points that are centered at the
                              mid-vertebral levels. Separate desired levels with ",". To get all
                              levels, enter "0".
 -vert-continuous             Convert discrete vertebral labeling to continuous vertebral labeling.
 -MSE <file>                  Compute Mean Square Error between labels from input and reference
                              image. Specify reference image here.
 -remove-reference <file>     Remove labels from input image (-i) that are not in reference image
                              (specified here).
 -remove-sym <file>           Remove labels from input image (-i) and reference image (specified
                              here) that don't match. You must provide two output names separated
                              by ",".
 -remove <list of: int>       Remove labels of specific value (specified here) from reference image
 -keep <list of: int>         Keep labels of specific value (specified here) from reference image
MISC
 -msg <str>                   Display a message to explain the labeling task. Use with
                              -create-viewer.
 -o <list of: file_output>    Output image(s). Default value = labels.nii.gz
 -v {0,1,2}                   Verbose. 0: nothing. 1: basic. 2: extended. Default value = 1
"""


class SCTLabelUtils_InputSpec(CommandLineInputSpec):
    in_file = File(exists=True, argstr="-i \"%s\"",
                   mandatory=True, position=0, desc="input file")
    create_viewer = traits.Str(
        exists=True, argstr="-create-viewer \"%s\"", mandatory=False, desc="list")
    keep = traits.Str(argstr="-keep %s", desc="keep")
    removesym_files = InputMultiPath(exists=True, argstr="-remove-sym \"%s\"",
                       mandatory=False, desc="remove-sym", sep=",")
    removereference_file = File(exists=True, argstr="-remove-reference \"%s\"",
                      mandatory=False, desc="remove-reference")


class SCTLabelUtils_OutputSpec(TraitedSpec):
    labels_file = File(exists=False)


class SCTLabelUtils(CommandLine):
    _cmd = "sct_label_utils "
    input_spec = SCTLabelUtils_InputSpec
    output_spec = SCTLabelUtils_OutputSpec

    def _run_interface(self, runtime, correct_return_codes=(0,)):
       runtime = super(SCTLabelUtils, self)._run_interface(runtime)
       return runtime

    def aggregate_outputs(self, runtime=None, needed_outputs=None):
       outputs = super(SCTLabelUtils, self).aggregate_outputs(
           runtime=runtime, needed_outputs=needed_outputs
       )
       return outputs

    def _list_outputs(self):
       outputs = self.output_spec().get()
       outputs['labels_file'] = os.path.abspath('labels.nii.gz')
       return outputs

       """
--
Spinal Cord Toolbox (5.2.0)

sct_maths
--

sct_maths: error: the following arguments are required: -i, -o

usage: sct_maths -i <file> -o <file> [-h] [-add  [...]] [-sub  [...]] [-mul  [...]] [-div  [...]] [-mean {x,y,z,t}] [-rms {x,y,z,t}] [-std {x,y,z,t}]
                 [-bin <float>] [-otsu <int>] [-adap <list>] [-otsu-median <list>] [-percent <int>] [-thr <float>] [-dilate <int>] [-erode <int>]
                 [-shape {square,cube,disk,ball}] [-dim {0,1,2}] [-smooth <list>] [-laplacian <list>] [-denoise DENOISE] [-mi <file>] [-minorm <file>]
                 [-corr <file>] [-symmetrize {0,1,2}] [-type {uint8,int16,int32,float32,complex64,float64,int8,uint16,uint32,int64,uint64}] [-v <int>]

Perform mathematical operations on images. Some inputs can be either a number or a 4d image or several 3d images separated with ","

MANDATORY ARGUMENTS:
  -i <file>             Input file. Example: data.nii.gz
  -o <file>             Output file. Example: data_mean.nii.gz

OPTIONAL ARGUMENTS:
  -h, --help            Show this help message and exit
  -v <int>              Verbosity. 0: Display only errors/warnings, 1: Errors/warnings + info messages, 2: Debug mode (default: 1)

BASIC OPERATIONS:
  -add  [ ...]          Add following input. Can be a number or multiple images (separated with space).
  -sub  [ ...]          Subtract following input. Can be a number or an image.
  -mul  [ ...]          Multiply by following input. Can be a number or multiple images (separated with space).
  -div  [ ...]          Divide by following input. Can be a number or an image.
  -mean {x,y,z,t}       Average data across dimension.
  -rms {x,y,z,t}        Compute root-mean-squared across dimension.
  -std {x,y,z,t}        Compute STD across dimension.
  -bin <float>          Binarize image using specified threshold. Example: 0.5

THRESHOLDING METHODS:
  -otsu <int>           Threshold image using Otsu algorithm (from skimage). Specify the number of bins (e.g. 16, 64, 128)
  -adap <list>          Threshold image using Adaptive algorithm (from skimage). Provide 2 values separated by ',' that correspond to the parameters below.
                        For example, '-adap 7,0' corresponds to a block size of 7 and an offset of 0.
                          - Block size: Odd size of pixel neighborhood which is used to calculate the threshold value.
                          - Offset: Constant subtracted from weighted mean of neighborhood to calculate the local threshold value. Suggested offset is 0.
  -otsu-median <list>   Threshold image using Median Otsu algorithm (from dipy). Provide 2 values separated by ',' that correspond to the parameters below.
                        For example, '-otsu-median 3,5' corresponds to a filter size of 3 repeated over 5 iterations.
                          - Size: Radius (in voxels) of the applied median filter.
                          - Iterations: Number of passes of the median filter.
  -percent <int>        Threshold image using percentile of its histogram.
  -thr <float>          Use following number to threshold image (zero below number).

MATHEMATICAL MORPHOLOGY:
  -dilate <int>         Dilate binary or greyscale image with specified size. If shape={'square', 'cube'}: size corresponds to the length of an edge (size=1
                        has no effect). If shape={'disk', 'ball'}: size corresponds to the radius, not including the center element (size=0 has no effect).
  -erode <int>          Erode binary or greyscale image with specified size. If shape={'square', 'cube'}: size corresponds to the length of an edge (size=1
                        has no effect). If shape={'disk', 'ball'}: size corresponds to the radius, not including the center element (size=0 has no effect).
  -shape {square,cube,disk,ball}
                        Shape of the structuring element for the mathematical morphology operation. Default: ball.
                        If a 2D shape {'disk', 'square'} is selected, -dim must be specified. (default: ball)
  -dim {0,1,2}          Dimension of the array which 2D structural element will be orthogonal to. For example, if you wish to apply a 2D disk kernel in the
                        X-Y plane, leaving Z unaffected, parameters will be: shape=disk, dim=2.

FILTERING METHODS:
  -smooth <list>        Gaussian smoothing filtering. Supply values for standard deviations in mm. If a single value is provided, it will be applied to each
                        axis of the image. If multiple values are provided, there must be one value per image axis. (Examples: "-smooth 2.0,3.0,2.0" (3D
                        image), "-smooth 2.0" (any-D image)).
  -laplacian <list>     Laplacian filtering. Supply values for standard deviations in mm. If a single value is provided, it will be applied to each axis of
                        the image. If multiple values are provided, there must be one value per image axis. (Examples: "-laplacian 2.0,3.0,2.0" (3D image),
                        "-laplacian 2.0" (any-D image)).
  -denoise DENOISE      Non-local means adaptative denoising from P. Coupe et al. as implemented in dipy. Separate with ". Example: p=1,b=3
                         p: (patch radius) similar patches in the non-local means are searched for locally, inside a cube of side 2*p+1 centered at each
                         voxel of interest. Default: p=1
                         b: (block radius) the size of the block to be used (2*b+1) in the blockwise non-local means implementation. Default: b=5     Note,
                         block radius must be smaller than the smaller image dimension: default value is lowered for small images)
                        To use default parameters, write -denoise 1

SIMILARITY METRIC:
  -mi <file>            Compute the mutual information (MI) between both input files (-i and -mi) as in: http://scikit-
                        learn.org/stable/modules/generated/sklearn.metrics.mutual_info_score.html
  -minorm <file>        Compute the normalized mutual information (MI) between both input files (-i and -mi) as in: http://scikit-
                        learn.org/stable/modules/generated/sklearn.metrics.normalized_mutual_info_score.html
  -corr <file>          Compute the cross correlation (CC) between both input files (-i and -cc).

MISC:
  -symmetrize {0,1,2}   Symmetrize data along the specified dimension.
  -type {uint8,int16,int32,float32,complex64,float64,int8,uint16,uint32,int64,uint64}
  """


class SCTMaths_InputSpec(CommandLineInputSpec):
    in_file = File(exists=True, argstr="-i \"%s\"",
                   mandatory=True, position=0, desc="input file")
    mean = traits.Str(argstr="-mean \"%s\"", mandatory=False, desc="mean")
    thr = traits.Str(argstr="-thr \"%s\"", mandatory=False, desc="thr")
    mul = traits.Str(argstr="-mul \"%s\"", mandatory=False, desc="mul")
    sub = traits.Str(argstr="-sub \"%s\"", mandatory=False, desc="sub")
    out_file = File(genfile=True, position=-2, argstr="-o \"%s\"",
                    desc="image to write", hash_files=False)


class SCTMaths_OutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="out")


class SCTMaths(CommandLine):
    _cmd = "sct_maths "
    input_spec = SCTMaths_InputSpec
    output_spec = SCTMaths_OutputSpec
    _suffix = "_sctmaths"

    def _run_interface(self, runtime, correct_return_codes=(0,)):
       runtime = super(SCTMaths, self)._run_interface(runtime)
       return runtime

    def aggregate_outputs(self, runtime=None, needed_outputs=None):
       outputs = super(SCTMaths, self).aggregate_outputs(
           runtime=runtime, needed_outputs=needed_outputs
       )
       return outputs

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if not isdefined(self.inputs.out_file):
            outputs["out_file"] = self._gen_fname(
                self.inputs.in_file, suffix=self._suffix
            )
        print("Output file: "+outputs["out_file"])
        outputs["out_file"] = os.path.abspath(outputs["out_file"])
        return outputs

    def _gen_filename(self, name):
        if name == "out_file":
            return self._list_outputs()["out_file"]
        return None

    def _gen_fname(self, basename, cwd=None, suffix=None, change_ext=True, ext=None):
        """Generate a filename based on the given parameters.
        The filename will take the form: cwd/basename<suffix><ext>.
        If change_ext is True, it will use the extentions specified in
        <instance>intputs.output_type.
        Parameters
        ----------
        basename : str
            Filename to base the new filename on.
        cwd : str
            Path to prefix to the new filename. (default is os.getcwd())
        suffix : str
            Suffix to add to the `basename`.  (defaults is '' )
        Returns
        -------
        fname : str
            New filename based on given parameters.
        """

        if basename == "":
            msg = "Unable to generate filename for command %s. " % self.cmd
            msg += "basename is not set!"
            raise ValueError(msg)
        if cwd is None:
            cwd = os.getcwd()
        if ext is None:
            ext = ".nii.gz"
        if change_ext:
            if suffix:
                suffix = "".join((suffix, ext))
            else:
                suffix = ext
        if suffix is None:
            suffix = ""
        fname = fname_presuffix(
            basename, suffix=suffix, use_ext=False, newpath=cwd)
        return fname


"""
--
Spinal Cord Toolbox (5.2.0)

sct_image
--

sct_image: error: the following arguments are required: -i

usage: sct_image -i <file> [<file> ...] [-h] [-o <file>] [-pad <list>] [-pad-asym <list>] [-split {x,y,z,t}] [-concat {x,y,z,t}] [-remove-vol <list>] [-keep-vol <list>]
                 [-type {uint8,int16,int32,float32,complex64,float64,int8,uint16,uint32,int64,uint64}] [-copy-header <file>] [-set-sform-to-qform] [-getorient]
                 [-setorient {RIP,LIP,RSP,LSP,RIA,LIA,RSA,LSA,IRP,ILP,SRP,SLP,IRA,ILA,SRA,SLA,RPI,LPI,RAI,LAI,RPS,LPS,RAS,LAS,PRI,PLI,ARI,ALI,PRS,PLS,ARS,ALS,IPR,SPR,IAR,SAR,IPL,SPL,IAL,SAL,PIR,PSR,AIR,ASR,PIL,PSL,AIL,ASL}]
                 [-setorient-data {RIP,LIP,RSP,LSP,RIA,LIA,RSA,LSA,IRP,ILP,SRP,SLP,IRA,ILA,SRA,SLA,RPI,LPI,RAI,LAI,RPS,LPS,RAS,LAS,PRI,PLI,ARI,ALI,PRS,PLS,ARS,ALS,IPR,SPR,IAR,SAR,IPL,SPL,IAL,SAL,PIR,PSR,AIR,ASR,PIL,PSL,AIL,ASL}]
                 [-mcs] [-omc] [-display-warp] [-to-fsl [<file> [<file> ...]]] [-v <int>]

Perform manipulations on images (e.g., pad, change space, split along dimension). Inputs can be a number, a 4d image, or several 3d images separated with ","

MANDATORY ARGUMENTS:
  -i <file> [<file> ...]
                        Input file(s). Example: "data.nii.gz"
                        Note: Only "-concat" or "-omc" support multiple input files. In those cases, separate filenames using spaces. Example usage: "sct_image -i data1.nii.gz data2.nii.gz -concat"

OPTIONAL ARGUMENTS:
  -h, --help            Show this help message and exit
  -o <file>             Output file. Example: data_pad.nii.gz

IMAGE OPERATIONS:
  -pad <list>           Pad 3D image. Specify padding as: "x,y,z" (in voxel). Example: "0,0,1"
  -pad-asym <list>      Pad 3D image with asymmetric padding. Specify padding as: "x_i,x_f,y_i,y_f,z_i,z_f" (in voxel). Example: "0,0,5,10,1,1"
  -split {x,y,z,t}      Split data along the specified dimension. The suffix _DIM+NUMBER will be added to the intput file name.
  -concat {x,y,z,t}     Concatenate data along the specified dimension
  -remove-vol <list>    Remove specific volumes from a 4d volume. Separate with ",". Example: "0,5,10"
  -keep-vol <list>      Keep specific volumes from a 4d volume (remove others). Separate with ",". Example: "1,2,3,11"
  -type {uint8,int16,int32,float32,complex64,float64,int8,uint16,uint32,int64,uint64}
                        Change file type

HEADER OPERATIONS:
  -copy-header <file>   Copy the header of the source image (specified in -i) to the destination image (specified here) and save it into a new image (specified in -o)
  -set-sform-to-qform   Set the input image's sform matrix equal to its qform matrix. Use this option when you need to enforce matching sform and qform matrices. This option can be used by itself, or
                        in combination with other functions. (default: False)

ORIENTATION OPERATIONS:
  -getorient            Get orientation of the input image (default: False)
  -setorient {RIP,LIP,RSP,LSP,RIA,LIA,RSA,LSA,IRP,ILP,SRP,SLP,IRA,ILA,SRA,SLA,RPI,LPI,RAI,LAI,RPS,LPS,RAS,LAS,PRI,PLI,ARI,ALI,PRS,PLS,ARS,ALS,IPR,SPR,IAR,SAR,IPL,SPL,IAL,SAL,PIR,PSR,AIR,ASR,PIL,PSL,AIL,ASL}
                        Set orientation of the input image (only modifies the header).
  -setorient-data {RIP,LIP,RSP,LSP,RIA,LIA,RSA,LSA,IRP,ILP,SRP,SLP,IRA,ILA,SRA,SLA,RPI,LPI,RAI,LAI,RPS,LPS,RAS,LAS,PRI,PLI,ARI,ALI,PRS,PLS,ARS,ALS,IPR,SPR,IAR,SAR,IPL,SPL,IAL,SAL,PIR,PSR,AIR,ASR,PIL,PSL,AIL,ASL}
                        Set orientation of the input image's data (does NOT modify the header, but the data). Use with care !

MULTI-COMPONENT OPERATIONS ON ITK COMPOSITE WARPING FIELDS:
  -mcs                  Multi-component split: Split ITK warping field into three separate displacement fields. The suffix _X, _Y and _Z will be added to the input file name. (default: False)
  -omc                  Multi-component merge: Merge inputted images into one multi-component image. Requires several inputs. (default: False)

WARPING FIELD OPERATIONS::
  -display-warp         Create a grid and deform it using provided warping field. (default: False)
  -to-fsl [<file> [<file> ...]]
                        Transform displacement field values to absolute FSL warps. To be used with FSL's applywarp function with the `--abs` flag. Input the file that will be used as the input
                        (source) for applywarp and optionally the target (ref). The target file is necessary for the case where the warp is in a different space than the target. For example, the
                        inverse warps generated by `sct_straighten_spinalcord`. This feature has not been extensively validated so consider checking the results of `applywarp` against
                        `sct_apply_transfo` before using in FSL pipelines. Example syntax: "sct_image -i WARP_SRC2DEST -to-fsl IM_SRC (IM_DEST) -o WARP_FSL", followed by FSL: "applywarp -i IM_SRC -r
                        IM_DEST -w WARP_FSL --abs -o IM_SRC2DEST"

Misc:
  -v <int>              Verbosity. 0: Display only errors/warnings, 1: Errors/warnings + info messages, 2: Debug mode (default: 1)
"""


class SCTImage_InputSpec(CommandLineInputSpec):
    in_file = File(exists=True, argstr="-i \"%s\"",
                   mandatory=True, position=0, desc="input file")
    pad = traits.Str(argstr="-pad \"%s\"", mandatory=False, desc="pad")
    setorient = traits.Str(argstr="-setorient %s",
                           mandatory=False, desc="setorient")
    setorientdata = traits.Str(
        argstr="-setorient-data %s", mandatory=False, desc="setorient-data")
    setsformtoqform = traits.Bool(
        argstr="-set-sform-to-qform", mandatory=False, desc="setsform2qform")
    out_file = File(genfile=True, argstr="-o \"%s\"",
                    desc="output file", hash_files=False)


class SCTImage_OutputSpec(TraitedSpec):
    out_file = File(exists=False, desc="out")


class SCTImage(CommandLine):
    _cmd = "sct_image "
    input_spec = SCTImage_InputSpec
    output_spec = SCTImage_OutputSpec
    _suffix = "_sctimage"

    def _run_interface(self, runtime, correct_return_codes=(0,)):
       runtime = super(SCTImage, self)._run_interface(runtime)
       return runtime

    def aggregate_outputs(self, runtime=None, needed_outputs=None):
       outputs = super(SCTImage, self).aggregate_outputs(
           runtime=runtime, needed_outputs=needed_outputs
       )
       return outputs

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if not isdefined(self.inputs.out_file):
            outputs["out_file"] = self._gen_fname(
                self.inputs.in_file, suffix=self._suffix
            )
        print("Output file: "+outputs["out_file"])
        outputs["out_file"] = os.path.abspath(outputs["out_file"])
        return outputs

    def _gen_filename(self, name):
        if name == "out_file":
            return self._list_outputs()["out_file"]
        return None

    def _gen_fname(self, basename, cwd=None, suffix=None, change_ext=True, ext=None):
        """
        Generate a filename based on the given parameters.
        The filename will take the form: cwd/basename<suffix><ext>.
        If change_ext is True, it will use the extentions specified in
        <instance>intputs.output_type.
        Parameters
        ----------
        basename : str
            Filename to base the new filename on.
        cwd : str
            Path to prefix to the new filename. (default is os.getcwd())
        suffix : str
            Suffix to add to the `basename`.  (defaults is '' )
        Returns
        -------
        fname : str
            New filename based on given parameters.
        """

        if basename == "":
            msg = "Unable to generate filename for command %s. " % self.cmd
            msg += "basename is not set!"
            raise ValueError(msg)
        if cwd is None:
            cwd = os.getcwd()
        if ext is None:
            ext = ".nii.gz"
        if change_ext:
            if suffix:
                suffix = "".join((suffix, ext))
            else:
                suffix = ext
        if suffix is None:
            suffix = ""
        fname = fname_presuffix(
            basename, suffix=suffix, use_ext=False, newpath=cwd)
        return fname


"""
--
Spinal Cord Toolbox (5.2.0)

sct_fmri_moco
--

sct_fmri_moco: error: the following arguments are required: -i

usage: sct_fmri_moco -i <file> [-h] [-g <int>] [-m <file>] [-param <list>] [-ofolder <folder>] [-x {nn,linear,spline}] [-r <int>] [-v <int>]

Motion correction of fMRI data. Some robust features include:
  - group-wise (-g)
  - slice-wise regularized along z using polynomial function (-p)
    (For more info about the method, type: isct_antsSliceRegularizedRegistration)
  - masking (-m)
  - iterative averaging of target volume

The outputs of the motion correction process are:
  - the motion-corrected fMRI volumes
  - the time average of the corrected fMRI volumes
  - a time-series with 1 voxel in the XY plane, for the X and Y motion direction (two separate files), as required for FSL analysis.
  - a TSV file with the slice-wise average of the motion correction for XY (one file), that can be used for Quality Control.

MANDATORY ARGUMENTS:
  -i <file>             Input data (4D). Example: fmri.nii.gz

OPTIONAL ARGUMENTS:
  -h, --help            Show this help message and exit.
  -g <int>              Group nvols successive fMRI volumes for more robustness.
  -m <file>             Binary mask to limit voxels considered by the registration metric.
  -param <list>         Advanced parameters. Assign value with "="; Separate arguments with ",".
                          - poly [int]: Degree of polynomial function used for regularization along Z. For no regularization set to 0. Default=2.
                          - smooth [mm]: Smoothing kernel. Default=0.
                          - iter [int]: Number of iterations. Default=10.
                          - metric {MI, MeanSquares, CC}: Metric used for registration. Default=MeanSquares.
                          - gradStep [float]: Searching step used by registration algorithm. The higher the more deformation allowed. Default=1.
                          - sampling [None or 0-1]: Sampling rate used for registration metric. Default=None.
                          - numTarget [int]: Target volume or group (starting with 0). Default=0.
                          - iterAvg [int]: Iterative averaging: Target volume is a weighted average of the previously-registered volumes. Default=1.
  -ofolder <folder>     Output path. (default: ./)
  -x {nn,linear,spline}
                        Final interpolation. (default: linear)
  -r <int>              Remove temporary files. O = no, 1 = yes (default: 1)
  -v <int>              Verbosity. 0: Display only errors/warnings, 1: Errors/warnings + info messages, 2: Debug mode (default: 1)

"""


class SCTFmriMoco_InputSpec(CommandLineInputSpec):
    in_file = File(exists=True, argstr="-i \"%s\" ",
                   mandatory=True, position=1, desc="input file")
    param = traits.Str(argstr="-param \"%s\" ",
                       mandatory=False, desc="param list")


class SCTFmriMoco_OutputSpec(TraitedSpec):
    out_file = File(exists=False)
    meanout_file = File(exists=False)


class SCTFmriMoco(CommandLine):
    _cmd = "sct_fmri_moco -ofolder . "
    input_spec = SCTFmriMoco_InputSpec
    output_spec = SCTFmriMoco_OutputSpec
    _suffix = "_moco"
    _suffixmean = "_moco_mean"

    def _run_interface(self, runtime, correct_return_codes=(0,)):
       runtime = super(SCTFmriMoco, self)._run_interface(runtime)
       return runtime

    def aggregate_outputs(self, runtime=None, needed_outputs=None):
       outputs = super(SCTFmriMoco, self).aggregate_outputs(
           runtime=runtime, needed_outputs=needed_outputs
       )
       return outputs

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["out_file"] = self._gen_fname(
            self.inputs.in_file, suffix=self._suffix, ext='.nii')
        outputs["meanout_file"] = self._gen_fname(
            self.inputs.in_file, suffix=self._suffixmean, ext='.nii')
        print("Output file: "+outputs["out_file"])
        outputs["out_file"] = os.path.abspath(outputs["out_file"])
        outputs["meanout_file"] = os.path.abspath(outputs["meanout_file"])
        return outputs

    def _gen_fname(self, basename, cwd=None, suffix=None, change_ext=True, ext=None):
        """Generate a filename based on the given parameters.
        The filename will take the form: cwd/basename<suffix><ext>.
        If change_ext is True, it will use the extentions specified in
        <instance>intputs.output_type.
        Parameters
        ----------
        basename : str
            Filename to base the new filename on.
        cwd : str
            Path to prefix to the new filename. (default is os.getcwd())
        suffix : str
            Suffix to add to the `basename`.  (defaults is '' )
        Returns
        -------
        fname : str
            New filename based on given parameters.
        """

        if basename == "":
            msg = "Unable to generate filename for command %s. " % self.cmd
            msg += "basename is not set!"
            raise ValueError(msg)
        if cwd is None:
            cwd = os.getcwd()
        if ext is None:
            ext = ".nii.gz"
        if change_ext:
            if suffix:
                suffix = "".join((suffix, ext))
            else:
                suffix = ext
        if suffix is None:
            suffix = ""
        fname = fname_presuffix(
            basename, suffix=suffix, use_ext=False, newpath=cwd)
        return fname


"""
--
Spinal Cord Toolbox (5.2.0)

sct_crop_image
--

sct_crop_image: error: the following arguments are required: -i

usage: sct_crop_image -i <file> [-h] [-o <str>] [-g {0,1}] [-m <file>] [-ref <file>] [-xmin <int>] [-xmax <int>] [-ymin <int>] [-ymax <int>] [-zmin <int>] [-zmax <int>] [-b <int>] [-v <int>]

Tools to crop an image. Either via command line or via a Graphical User Interface (GUI). See example usage at the end.

MANDATORY ARGUMENTS:
  -i <file>    Input image. Example: t2.nii.gz

OPTIONAL ARGUMENTS:
  -h, --help   Show this help message and exit
  -o <str>     Output image. By default, the suffix '_crop' will be added to the input image.
  -g {0,1}     0: Cropping via command line | 1: Cropping via GUI. Has priority over -m. (default: 0)
  -m <file>    Binary mask that will be used to extract bounding box for cropping the image. Has priority over -ref.
  -ref <file>  Image which dimensions (in the physical coordinate system) will be used as a reference to crop the input image. Only works for 3D images. Has priority over min/max method.
  -xmin <int>  Lower bound for cropping along X. (default: 0)
  -xmax <int>  Higher bound for cropping along X. Setting '-1' will crop to the maximum dimension (i.e. no change), '-2' will crop to the maximum dimension minus 1 slice, etc. (default: -1)
  -ymin <int>  Lower bound for cropping along Y. (default: 0)
  -ymax <int>  Higher bound for cropping along Y. Follows the same rules as xmax. (default: -1)
  -zmin <int>  Lower bound for cropping along Z. (default: 0)
  -zmax <int>  Higher bound for cropping along Z. Follows the same rules as xmax. (default: -1)
  -b <int>     If this flag is declared, the image will not be cropped (i.e. the dimension will not change). Instead, voxels outside the bounding box will be set to the value specified by this flag.
               For example, to have zeros outside the bounding box, use: '-b 0'
  -v <int>     Verbosity. 0: Display only errors/warnings, 1: Errors/warnings + info messages, 2: Debug mode (default: 1)

EXAMPLES:
- To crop an image using the GUI (this does not allow to crop along the right-left dimension):
sct_crop_image -i t2.nii.gz -g 1

- To crop an image using a binary mask:
sct_crop_image -i t2.nii.gz -m mask.nii.gz

- To crop an image using a reference image:
sct_crop_image -i t2.nii.gz -ref mt1.nii.gz

- To crop an image by specifying min/max (you don't need to specify all dimensions). In the example below, cropping will occur between x=5 and x=60, and between z=5 and z=zmax-1
sct_crop_image -i t2.nii.gz -xmin 5 -xmax 60 -zmin 5 -zmax -2
"""


class SCTCrop_InputSpec(CommandLineInputSpec):
    in_file = File(exists=True, argstr="-i \"%s\"",
                   mandatory=True, position=0, desc="input file")
    mask = File(exists=True, argstr="-m %s", mandatory=False, desc="mask")
    ref = File(exists=True, argstr="-ref %s", mandatory=False, desc="ref")
    xmin = traits.Str(argstr="-xmin %s", mandatory=False, desc="xmin")
    xmax = traits.Str(argstr="-xmax %s", mandatory=False, desc="xmax")
    ymin = traits.Str(argstr="-ymin %s", mandatory=False, desc="ymin")
    ymax = traits.Str(argstr="-ymax %s", mandatory=False, desc="ymax")
    zmin = traits.Str(argstr="-zmin %s", mandatory=False, desc="zmin")
    zmax = traits.Str(argstr="-zmax %s", mandatory=False, desc="zmax")
    b = traits.Str(argstr="-b %d", mandatory=False, desc="bouding box val")
    out_file = File(genfile=True, argstr="-o \"%s\"",
                    desc="output file", hash_files=False)


class SCTCrop_OutputSpec(TraitedSpec):
    out_file = File(exists=False, desc="out")


class SCTCrop(CommandLine):
    _cmd = "sct_crop_image "
    input_spec = SCTCrop_InputSpec
    output_spec = SCTCrop_OutputSpec
    _suffix = "_crop"

    def _run_interface(self, runtime, correct_return_codes=(0,)):
       runtime = super(SCTCrop, self)._run_interface(runtime)
       return runtime

    def aggregate_outputs(self, runtime=None, needed_outputs=None):
       outputs = super(SCTCrop, self).aggregate_outputs(
           runtime=runtime, needed_outputs=needed_outputs
       )
       return outputs

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if not isdefined(self.inputs.out_file):
            outputs["out_file"] = self._gen_fname(
                self.inputs.in_file, suffix=self._suffix
            )
        print("Output file: "+outputs["out_file"])
        outputs["out_file"] = os.path.abspath(outputs["out_file"])
        return outputs

    def _gen_filename(self, name):
        if name == "out_file":
            return self._list_outputs()["out_file"]
        return None

    def _gen_fname(self, basename, cwd=None, suffix=None, change_ext=True, ext=None):
        """Generate a filename based on the given parameters.
        The filename will take the form: cwd/basename<suffix><ext>.
        If change_ext is True, it will use the extentions specified in
        <instance>intputs.output_type.
        Parameters
        ----------
        basename : str
            Filename to base the new filename on.
        cwd : str
            Path to prefix to the new filename. (default is os.getcwd())
        suffix : str
            Suffix to add to the `basename`.  (defaults is '' )
        Returns
        -------
        fname : str
            New filename based on given parameters.
        """

        if basename == "":
            msg = "Unable to generate filename for command %s. " % self.cmd
            msg += "basename is not set!"
            raise ValueError(msg)
        if cwd is None:
            cwd = os.getcwd()
        if ext is None:
            ext = ".nii.gz"
        if change_ext:
            if suffix:
                suffix = "".join((suffix, ext))
            else:
                suffix = ext
        if suffix is None:
            suffix = ""
        fname = fname_presuffix(
            basename, suffix=suffix, use_ext=False, newpath=cwd)
        return fname

"""
--
Spinal Cord Toolbox (5.2.0)

sct_process_segmentation
--

sct_process_segmentation: error: the following arguments are required: -i

usage: sct_process_segmentation -i <file> [-h] [-o <file>] [-append <int>] [-z <str>] [-perslice <int>] [-vert <str>] [-vertfile <str>] [-perlevel <int>] [-r <int>] [-angle-corr <int>]
                                [-centerline-algo {polyfit,bspline,linear,nurbs}] [-centerline-smooth <int>] [-qc <folder>] [-qc-dataset <str>] [-qc-subject <str>] [-v <int>]

Compute the following morphometric measures based on the spinal cord segmentation:
  - area [mm^2]: Cross-sectional area, measured by counting pixels in each slice. Partial volume can be accounted for by inputing a mask comprising values within [0,1].
  - angle_AP, angle_RL: Estimated angle between the cord centerline and the axial slice. This angle is used to correct for morphometric information.
  - diameter_AP, diameter_RL: Finds the major and minor axes of the cord and measure their length.
  - eccentricity: Eccentricity of the ellipse that has the same second-moments as the spinal cord. The eccentricity is the ratio of the focal distance (distance between focal points) over the major axis length. The value is in the interval [0, 1). When it is 0, the ellipse becomes a circle.
  - orientation: angle (in degrees) between the AP axis of the spinal cord and the AP axis of the image
  - solidity: CSA(spinal_cord) / CSA_convex(spinal_cord). If perfect ellipse, it should be one. This metric is interesting for detecting non-convex shape (e.g., in case of strong compression)
  - length: Length of the segmentation, computed by summing the slice thickness (corrected for the centerline angle at each slice) across the specified superior-inferior region.

MANDATORY ARGUMENTS:
  -i <file>             Mask to compute morphometrics from. Could be binary or weighted. E.g., spinal cord segmentation.Example: seg.nii.gz

OPTIONAL ARGUMENTS:
  -h, --help            Show this help message and exit.
  -o <file>             Output file name (add extension). Default: csa.csv.
  -append <int>         Append results as a new line in the output csv file instead of overwriting it. (default: 0)
  -z <str>              Slice range to compute the metrics across (requires '-p csa'). Example: 5:23
  -perslice <int>       Set to 1 to output one metric per slice instead of a single output metric. Please note that when methods ml or map is used, outputing a single metric per slice and then
                        averaging them all is not the same as outputting a single metric at once across all slices. (default: 0)
  -vert <str>           Vertebral levels to compute the metrics across. Example: 2:9 for C2 to T2.
  -vertfile <str>       Vertebral labeling file. Only use with flag -vert.
                        The input and the vertebral labelling file must in the same voxel coordinate system and must match the dimensions between each other.  (default:
                        ./label/template/PAM50_levels.nii.gz)
  -perlevel <int>       Set to 1 to output one metric per vertebral level instead of a single output metric. This flag needs to be used with flag -vert. (default: 0)
  -r <int>              Removes temporary folder used for the algorithm at the end of execution. (default: 1)
  -angle-corr <int>     Angle correction for computing morphometric measures. When angle correction is used, the cord within the slice is stretched/expanded by a factor corresponding to the cosine of
                        the angle between the centerline and the axial plane. If the cord is already quasi-orthogonal to the slab, you can set -angle-corr to 0. (default: 1)
  -centerline-algo {polyfit,bspline,linear,nurbs}
                        Algorithm for centerline fitting. Only relevant with -angle-corr 1. (default: bspline)
  -centerline-smooth <int>
                        Degree of smoothing for centerline fitting. Only use with -centerline-algo {bspline, linear}. (default: 30)
  -qc <folder>          The path where the quality control generated content will be saved.
  -qc-dataset <str>     If provided, this string will be mentioned in the QC report as the dataset the process was run on.
  -qc-subject <str>     If provided, this string will be mentioned in the QC report as the subject the process was run on.
  -v <int>              Verbosity. 0: Display only errors/warnings, 1: Errors/warnings + info messages, 2: Debug mode (default: 1)
  """


class SCTProcessSegmentation_InputSpec(CommandLineInputSpec):
    in_file = File(exists=True, argstr="-i \"%s\"",
                   mandatory=True, position=0, desc="input file")
    perslice = traits.Str(argstr="-perslice %s",
                          mandatory=False, desc="perslice")
    vert = traits.Str(argstr="-vert %s", mandatory=False, desc="vert")
    vertfile = File(exists=True, argstr="-vertfile %s",
                    mandatory=False, desc="vertfile")
    z = traits.Str(argstr="-z %s", mandatory=False, desc="perslice")
    out_file = File(genfile=True, argstr="-o \"%s\"",
                    desc="output file", hash_files=False)


class SCTProcessSegmentation_OutputSpec(TraitedSpec):
    out_file = File(exists=False, desc="out")


class SCTProcessSegmentation(CommandLine):
    _cmd = "sct_process_segmentation "
    input_spec = SCTProcessSegmentation_InputSpec
    output_spec = SCTProcessSegmentation_OutputSpec
    _suffix = "_crop"

    def _run_interface(self, runtime, correct_return_codes=(0,)):
       runtime = super(SCTProcessSegmentation, self)._run_interface(runtime)
       return runtime

    def aggregate_outputs(self, runtime=None, needed_outputs=None):
       outputs = super(SCTProcessSegmentation, self).aggregate_outputs(
           runtime=runtime, needed_outputs=needed_outputs
       )
       return outputs

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if not isdefined(self.inputs.out_file):
            outputs["out_file"] = self._gen_fname(
                self.inputs.in_file, suffix=self._suffix
            )
        print("Output file: "+outputs["out_file"])
        outputs["out_file"] = os.path.abspath(outputs["out_file"])
        return outputs

    def _gen_filename(self, name):
        if name == "out_file":
            return self._list_outputs()["out_file"]
        return None

    def _gen_fname(self, basename, cwd=None, suffix=None, change_ext=True, ext=None):
        """Generate a filename based on the given parameters.
        The filename will take the form: cwd/basename<suffix><ext>.
        If change_ext is True, it will use the extentions specified in
        <instance>intputs.output_type.
        Parameters
        ----------
        basename : str
            Filename to base the new filename on.
        cwd : str
            Path to prefix to the new filename. (default is os.getcwd())
        suffix : str
            Suffix to add to the `basename`.  (defaults is '' )
        Returns
        -------
        fname : str
            New filename based on given parameters.
        """

        if basename == "":
            msg = "Unable to generate filename for command %s. " % self.cmd
            msg += "basename is not set!"
            raise ValueError(msg)
        if cwd is None:
            cwd = os.getcwd()
        if ext is None:
            ext = ".csv"
        if change_ext:
            if suffix:
                suffix = "".join((suffix, ext))
            else:
                suffix = ext
        if suffix is None:
            suffix = ""
        fname = fname_presuffix(
            basename, suffix=suffix, use_ext=False, newpath=cwd)
        return fname


"""
--
Spinal Cord Toolbox (5.2.0)

sct_dmri_separate_b0_and_dwi
--

sct_dmri_separate_b0_and_dwi: error: the following arguments are required: -i, -bvec

usage: sct_dmri_separate_b0_and_dwi -i <file> -bvec <file> [-h] [-a {0,1}] [-bval <file>] [-bvalmin <float>] [-ofolder <folder>] [-r {0,1}] [-v <int>]

Separate b=0 and DW images from diffusion dataset. The output files will have a suffix (_b0 and _dwi) appended to the input file name.

MANDATORY ARGUMENTS:
  -i <file>          Diffusion data. Example: dmri.nii.gz
  -bvec <file>       Bvecs file. Example: bvecs.txt

OPTIONAL ARGUMENTS:
  -h, --help         Show this help message and exit.
  -a {0,1}           Average b=0 and DWI data. 0 = no, 1 = yes (default: 1)
  -bval <file>       bvals file. Used to identify low b-values (in case different from 0). Example: bvals.nii.gz
  -bvalmin <float>   B-value threshold (in s/mm2) below which data is considered as b=0. Example: 50.0
  -ofolder <folder>  Output folder. Example: dmri_separate_results/ (default: ./)
  -r {0,1}           Remove temporary files. 0 = no, 1 = yes (default: 1)
  -v <int>           Verbosity. 0: Display only errors/warnings, 1: Errors/warnings + info messages, 2: Debug mode (default: 1)
"""


class SCTSeparateB0DWI_InputSpec(CommandLineInputSpec):
    in_file = File(exists=True, argstr="-i \"%s\"",
                   mandatory=True, position=0, desc="input file")
    bvecs_file = File(exists=True, argstr="-bvec \"%s\"",
                     mandatory=True, desc="bvec file")
    bvals_file = File(exists=True, argstr="-bval \"%s\"",
                     mandatory=False, desc="bval file")
    bvalmin = traits.Str(argstr="-bvalmin \"%s\"",
                         mandatory=False,  desc="bvalmin")
    average = traits.Str(argstr="-a \"%s\"", mandatory=False,  desc="average")
    b0_file = File(genfile=True,  desc="output b0 file", hash_files=False)
    dwi_file = File(genfile=True, desc="output dwi file", hash_files=False)


class SCTSeparateB0DWI_OutputSpec(TraitedSpec):
    b0_file = File(exists=False, desc="b0")
    dwi_file = File(exists=False, desc="dwi")


class SCTSeparateB0DWI(CommandLine):
    _cmd = "sct_dmri_separate_b0_and_dwi "
    input_spec = SCTSeparateB0DWI_InputSpec
    output_spec = SCTSeparateB0DWI_OutputSpec
    _suffixb0 = "_b0_mean"
    _suffixdwi = "_dwi_mean"

    def _run_interface(self, runtime, correct_return_codes=(0,)):
       runtime = super(SCTSeparateB0DWI, self)._run_interface(runtime)
       return runtime

    def aggregate_outputs(self, runtime=None, needed_outputs=None):
       outputs = super(SCTSeparateB0DWI, self).aggregate_outputs(
           runtime=runtime, needed_outputs=needed_outputs
       )
       return outputs

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if not isdefined(self.inputs.b0_file):
            outputs["b0_file"] = self._gen_fname(
                self.inputs.in_file, suffix=self._suffixb0, change_ext=True, ext=".nii"
            )
        if not isdefined(self.inputs.dwi_file):
            outputs["dwi_file"] = self._gen_fname(
                self.inputs.in_file, suffix=self._suffixdwi, change_ext=True, ext=".nii"
            )
        print("Output b0 file: "+outputs["b0_file"])
        outputs["b0_file"] = os.path.abspath(outputs["b0_file"])
        outputs["dwi_file"] = os.path.abspath(outputs["dwi_file"])
        return outputs

    def _gen_filename(self, name):
        if name == "b0_file":
            return self._list_outputs()["b0_file"]
        if name == "dwi_file":
            return self._list_outputs()["dwi_file"]
        return None

    def _gen_fname(self, basename, cwd=None, suffix=None, change_ext=True, ext=None):
        """Generate a filename based on the given parameters.
        The filename will take the form: cwd/basename<suffix><ext>.
        If change_ext is True, it will use the extentions specified in
        <instance>intputs.output_type.
        Parameters
        ----------
        basename : str
            Filename to base the new filename on.
        cwd : str
            Path to prefix to the new filename. (default is os.getcwd())
        suffix : str
            Suffix to add to the `basename`.  (defaults is '' )
        Returns
        -------
        fname : str
            New filename based on given parameters.
        """

        if basename == "":
            msg = "Unable to generate filename for command %s. " % self.cmd
            msg += "basename is not set!"
            raise ValueError(msg)
        if cwd is None:
            cwd = os.getcwd()
        if ext is None:
            ext = ".nii.gz"
        if change_ext:
            if suffix:
                suffix = "".join((suffix, ext))
            else:
                suffix = ext
        if suffix is None:
            suffix = ""
        fname = fname_presuffix(
            basename, suffix=suffix, use_ext=False, newpath=cwd)
        return fname



"""
--
Spinal Cord Toolbox (5.2.0)

sct_concat_transfo
--

sct_concat_transfo: error: the following arguments are required: -d, -w

usage: sct_concat_transfo -d <file> -w <file> [<file> ...] [-winv [<file> [<file> ...]]] [-h] [-o <str>] [-v <int>]

Concatenate transformations. This function is a wrapper for isct_ComposeMultiTransform (ANTs). The order of input warping fields is important. For example, if you want to concatenate: A->B and B->C to yield A->C, then you have to input warping fields in this order: A->B B->C.

MANDATORY ARGUMENTS:
  -d <file>             Destination image. (e.g. "mt.nii.gz")
  -w <file> [<file> ...]
                        Transformation(s), which can be warping fields (nifti image) or affine transformation matrix (text file). Separate with space. Example: warp1.nii.gz warp2.nii.gz

OPTIONAL ARGUMENTS:
  -winv [<file> [<file> ...]]
                        Affine transformation(s) listed in flag -w which should be inverted before being used. Note that this only concerns affine transformation (not warping fields). If you
                        would like to use an inverse warpingfield, then directly input the inverse warping field in flag -w.
  -h, --help            show this help message and exit
  -o <str>              Name of output warping field (e.g. "warp_template2mt.nii.gz")
  -v <int>              Verbosity. 0: Display only errors/warnings, 1: Errors/warnings + info messages, 2: Debug mode (default: 1)
  """


class SCTConcatTransform_InputSpec(CommandLineInputSpec):
    in_files = InputMultiPath(File(exists=True), argstr="-w %s ",
                   mandatory=True, desc="input file")
    dest_file = File(exists=False, mandatory=True, argstr="-d %s", desc="dest file")


class SCTConcatTransform_OutputSpec(TraitedSpec):
    warp_file = File(exists=False, desc="warp file")


class SCTConcatTransform(CommandLine):
    _cmd = "sct_concat_transfo "
    input_spec = SCTConcatTransform_InputSpec
    output_spec = SCTConcatTransform_OutputSpec

    def _run_interface(self, runtime, correct_return_codes=(0,)):
       runtime = super(SCTConcatTransform, self)._run_interface(runtime)
       return runtime

    def aggregate_outputs(self, runtime=None, needed_outputs=None):
       outputs = super(SCTConcatTransform, self).aggregate_outputs(
           runtime=runtime, needed_outputs=needed_outputs
       )
       return outputs

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["warp_file"] = os.path.abspath("warp_final.nii.gz")
        return outputs

"""
--
Spinal Cord Toolbox (dev)

sct_analyze_lesion -h
--

usage: sct_analyze_lesion -m <file> -s <file> [-h] [-i <file>] [-f <str>]
                          [-ofolder <folder>] [-r {0,1}] [-v <int>]

Compute statistics on segmented lesions. The function assigns an ID value to
each lesion (1, 2, 3, etc.) and then outputs morphometric measures for each
lesion:
- volume [mm^3]
- length [mm]: length along the Superior-Inferior axis
- max_equivalent_diameter [mm]: maximum diameter of the lesion, when
approximating
                                the lesion as a circle in the axial plane.

If the proportion of lesion in each region (e.g. WM and GM) does not sum up to
100%, it means that the registered template does not fully cover the lesion. In
that case you might want to check the registration results.

MANDATORY ARGUMENTS:
  -m <file>          Binary mask of lesions (lesions are labeled as "1").
  -s <file>          Spinal cord centerline or segmentation file, which will be
                     used to correct morphometric measures with cord angle with
                     respect to slice. (e.g.'t2_seg.nii.gz')

OPTIONAL ARGUMENTS:
  -h, --help         show this help message and exit
  -i <file>          Image from which to extract average values within lesions
                     (e.g. "t2.nii.gz"). If provided, the function computes the
                     mean and standard deviation values of this image within
                     each lesion.
  -f <str>           Path to folder containing the atlas/template registered to
                     the anatomical image. If provided, the function computes:
                     (i) the distribution of each lesion depending on each
                     vertebral level and on eachregion of the template (e.g. GM,
                     WM, WM tracts) and (ii) the proportion of ROI (e.g.
                     vertebral level, GM, WM) occupied by lesion.
  -ofolder <folder>  Output folder (e.g. "./") (default: ./)
  -r {0,1}           Remove temporary files. (default: 1)
  -v <int>           Verbosity. 0: Display only errors/warnings, 1:
                     Errors/warnings + info messages, 2: Debug mode (default: 1)
"""

class SCTAnalyzeLesion_InputSpec(CommandLineInputSpec):
    mask_file = File(exists=True, argstr="-m \"%s\"",
                   mandatory=True, position=0, desc="binary file")
    image_file = File(exists=False, argstr="-m \"%s\"",
                  mandatory=False, desc="input file")
    center_file = File(exists=True, argstr="-s \"%s\"",
                   mandatory=True, desc="seg file")
    xls_file = File(genfile=True,  desc="output xls file")
    pkl_file = File(genfile=True,  desc="output label file")
    label_file = File(genfile=True,  desc="output plk file")

class SCTAnalyzeLesion_OutputSpec(TraitedSpec):
    xls_file = File(exists=False, desc="xls")
    pkl_file = File(exists=False, desc="pkl")
    label_file = File(exists=False, desc="label")

class SCTAnalyzeLesion(CommandLine):
    _cmd = "sct_analyze_lesion "
    input_spec = SCTAnalyzeLesion_InputSpec
    output_spec = SCTAnalyzeLesion_OutputSpec

    def _run_interface(self, runtime, correct_return_codes=(0,)):
       runtime = super(SCTAnalyzeLesion, self)._run_interface(runtime)
       return runtime

    def aggregate_outputs(self, runtime=None, needed_outputs=None):
       outputs = super(SCTAnalyzeLesion, self).aggregate_outputs(
           runtime=runtime, needed_outputs=needed_outputs
       )
       return outputs

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if not isdefined(self.inputs.xls_file):
            outputs["xls_file"] = self._gen_fname(
                self.inputs.mask_file, suffix="_analyzis", ext=".xls"
            )
        if not isdefined(self.inputs.label_file):
            outputs["label_file"] = self._gen_fname(
                self.inputs.mask_file, suffix="_label", ext=".nii.gz"
            )
        if not isdefined(self.inputs.pkl_file):
            outputs["pkl_file"] = self._gen_fname(
                self.inputs.mask_file, suffix="_analyzis", ext=".plk"
            )
        print("Output file: "+outputs["xls_file"])
        outputs["xls_file"] = os.path.abspath(outputs["xls_file"])
        outputs["pkl_file"] = os.path.abspath(outputs["pkl_file"])
        outputs["label_file"] = os.path.abspath(outputs["label_file"])
        return outputs

    def _gen_filename(self, name):
        if name == "xls_file":
            return self._list_outputs()["xls_file"]
        if name == "pkl_file":
            return self._list_outputs()["pkl_file"]
        if name == "label_file":
            return self._list_outputs()["label_file"]
        return None

    def _gen_fname(self, basename, cwd=None, suffix=None, change_ext=True, ext=None):
        """Generate a filename based on the given parameters.
        The filename will take the form: cwd/basename<suffix><ext>.
        If change_ext is True, it will use the extentions specified in
        <instance>intputs.output_type.
        Parameters
        ----------
        basename : str
            Filename to base the new filename on.
        cwd : str
            Path to prefix to the new filename. (default is os.getcwd())
        suffix : str
            Suffix to add to the `basename`.  (defaults is '' )
        Returns
        -------
        fname : str
            New filename based on given parameters.
        """

        if basename == "":
            msg = "Unable to generate filename for command %s. " % self.cmd
            msg += "basename is not set!"
            raise ValueError(msg)
        if cwd is None:
            cwd = os.getcwd()
        if ext is None:
            ext = ".xls"
        if change_ext:
            if suffix:
                suffix = "".join((suffix, ext))
            else:
                suffix = ext
        if suffix is None:
            suffix = ""
        fname = fname_presuffix(
            basename, suffix=suffix, use_ext=False, newpath=cwd)
        return fname
