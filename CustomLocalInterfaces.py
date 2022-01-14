# Custom interfaces for DoD project data analysis (Budde)


from nipype.interfaces.base import BaseInterfaceInputSpec, BaseInterface, File, TraitedSpec, OutputMultiPath, traits, isdefined

import os
import nibabel as nib
import math
import numpy as np
from nibabel import load
from nipype.utils.filemanip import fname_presuffix

#from dipy.data import get_fnames
#from dipy.io.image import load_nifti, save_nifti

#from dipy.denoise.patch2self import patch2self
#from dipy.denoise.nlmeans import nlmeans
#from dipy.denoise.non_local_means import non_local_means

# Custom interface to remove XY rotation from images.
# This is used to improve registration by better aliginging the initial images.
# Since the cord can be rotated in animal positioning (compared to human),
# we typically align the slices to the cord anatomy as close as possible.
# this will get the Z rotation angle of an image (rot_file) and unrotate (*-1)
# a second image (in_file) and save it (out_file) with an updated sform/qform.
# the random number input is used to force the cache to redo this step.
#
#
#
class GenUnrotateMatrixInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True,  mandatory=True, position=0, desc="input file")
    rot_file = File(exists=True,  mandatory=True, position=1, desc="rotation file")
    randHack = traits.Str(mandatory=False, desc="rand ")
    out_file = File(exists=False,  mandatory=False, desc="out file")
    rot180 = traits.Bool(mandatory=False, desc="Additional Rotation") #this will implicitly default to

class  GenUnrotateMatrixtOutputSpec(TraitedSpec):
    out_file = File(exists=False)

class  GenUnrotateMatrix(BaseInterface):
    input_spec =  GenUnrotateMatrixInputSpec
    output_spec =  GenUnrotateMatrixtOutputSpec
    _suffix = "_unrot"

    def _run_interface(self, runtime):
        fname = self.inputs.in_file
        nii_img = nib.load(fname)
        header = nii_img.header
        sform = header.get_sform()
        print(sform)

        origmat = (sform[0:3,0:3])
        origtrans = (sform[0:3,3])
        print(origmat)
        print(origtrans)


        rot_fname = self.inputs.rot_file
        rot_nii_img = nib.load(rot_fname)
        rot_header = rot_nii_img.header
        rot_sform = rot_header.get_sform()
        print(rot_sform)

        rotangle = -1*(math.pi/2 + math.atan2(rot_sform[1,0],rot_sform[0,0]))
        print('Rot angle is ' + str(rotangle/math.pi*180))
        if (self.inputs.rot180 == True):
            rotangle = rotangle + math.pi
            print('Rotating additional 180 degrees')
        rotmat = [[math.cos(rotangle), math.sin(rotangle) * -1, 0], [math.sin(rotangle), math.cos(rotangle), 0],[0, 0, 1]]
        print(rotmat)


        newmat = np.matmul(rotmat, origmat)
        newtrans = np.matmul(rotmat, origtrans)
        print(newmat)
        print(newtrans)

        new_sform = sform
        new_sform[0:3, 0:3] = newmat
        new_sform[0:3, 3] = newtrans

        #header.set_sform(new_sform)
        #header.set_qform(new_sform)


        outfname = self._gen_fname(
            self.inputs.in_file, suffix=self._suffix
        )
        outimg = nib.Nifti1Image(np.asanyarray(nii_img.dataobj), new_sform, header)
        nii_img_out = nib.save(outimg, outfname)


        return runtime

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



#
# class Patch2SelfInputSpec(BaseInterfaceInputSpec):
#     in_file = File(exists=True,  mandatory=True, position=0, desc="input file")
#     bvals_file = File(exists=True,  mandatory=True, position=1, desc="bvals file")
#     randHack = traits.Str(mandatory=False, desc="rand ")
#     out_file = File(exists=False,  mandatory=False, desc="out file")
#
# class  Patch2SelfOutputSpec(TraitedSpec):
#     out_file = File(exists=False)
#
# class  Patch2Self(BaseInterface):
#     input_spec =  Patch2SelfInputSpec
#     output_spec =  Patch2SelfOutputSpec
#     _suffix = "_p2self"
#
#     def _run_interface(self, runtime):
#         fname = self.inputs.in_file
#         bvalfname = self.inputs.bvals_file
#
#         #hardi_fname, hardi_bval_fname, hardi_bvec_fname = get_fnames('stanford_hardi')
#         #data, affine = load_nifti(hardi_fname)
#         #bvals = np.loadtxt(hardi_bval_fname)
#         #denoised_arr = patch2self(data, bvals, model='ols', shift_intensity=True,
#         #                          clip_negative_vals=False, b0_threshold=50)
#
#         data, affine = load_nifti(fname)
#
#         print("Data:")
#         print(data.shape)
#         stackeddata = np.concatenate((data, data, data), axis=2)
#         print("Stacked Data:")
#         print(stackeddata.shape)
#
#         bvals = np.loadtxt(bvalfname)
#         denoised_arr = non_local_means(stackeddata, 0.5)
#         #denoised_arr = patch2self(stackeddata, bvals, model='ols', shift_intensity=False,
#         #                          clip_negative_vals=True, b0_denoising=True)
#         denoised_arr_stacked = denoised_arr[:,:,1,:]
#         denoised_arr_stacked = denoised_arr_stacked.reshape([data.shape[0],data.shape[1],1,data.shape[3]])
#
#         print("Denoised slice Data:")
#         print(denoised_arr_stacked.shape)
#         outfname = self._gen_fname(
#             self.inputs.in_file, suffix=self._suffix
#         )
#         save_nifti(outfname, denoised_arr_stacked, affine)
#
#         return runtime
#
#
#     def _list_outputs(self):
#         outputs = self.output_spec().get()
#         if not isdefined(self.inputs.out_file):
#             outputs["out_file"] = self._gen_fname(
#                 self.inputs.in_file, suffix=self._suffix
#             )
#         print("Output file: "+outputs["out_file"])
#         outputs["out_file"] = os.path.abspath(outputs["out_file"])
#         return outputs
#
#     def _gen_filename(self, name):
#         if name == "out_file":
#             return self._list_outputs()["out_file"]
#         return None
#
#     def _gen_fname(self, basename, cwd=None, suffix=None, change_ext=True, ext=None):
#         """Generate a filename based on the given parameters.
#         The filename will take the form: cwd/basename<suffix><ext>.
#         If change_ext is True, it will use the extentions specified in
#         <instance>intputs.output_type.
#         Parameters
#         ----------
#         basename : str
#             Filename to base the new filename on.
#         cwd : str
#             Path to prefix to the new filename. (default is os.getcwd())
#         suffix : str
#             Suffix to add to the `basename`.  (defaults is '' )
#         Returns
#         -------
#         fname : str
#             New filename based on given parameters.
#         """
#
#         if basename == "":
#             msg = "Unable to generate filename for command %s. " % self.cmd
#             msg += "basename is not set!"
#             raise ValueError(msg)
#         if cwd is None:
#             cwd = os.getcwd()
#         if ext is None:
#             ext = ".nii.gz"
#         if change_ext:
#             if suffix:
#                 suffix = "".join((suffix, ext))
#             else:
#                 suffix = ext
#         if suffix is None:
#             suffix = ""
#         fname = fname_presuffix(
#             basename, suffix=suffix, use_ext=False, newpath=cwd)
#         return fname

#
class GetFinalRegTargetInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True,  mandatory=True, position=0, desc="input file")
    alt_file = File(exists=False,  mandatory=False, position=0, desc="input file")

class  GetFinalRegTargetOutputSpec(TraitedSpec):
    target_file = File(exists=False)
    moving_file = File(exists=False)

class  GetFinalRegTarget(BaseInterface):
    input_spec =  GetFinalRegTargetInputSpec
    output_spec =  GetFinalRegTargetOutputSpec

    def _run_interface(self, runtime):

        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()

        if os.path.exists(self.inputs.alt_file):
            outputs["target_file"] = os.path.abspath(self.inputs.alt_file)
        else:
            outputs["target_file"] = os.path.abspath(self.inputs.in_file)

        outputs["moving_file"] = os.path.abspath(self.inputs.in_file)

        print("Output file: "+outputs["target_file"])
        return outputs
