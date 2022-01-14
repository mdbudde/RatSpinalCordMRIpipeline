from nipype.interfaces.base import (CommandLine, traits, TraitedSpec,
                                    BaseInterface, BaseInterfaceInputSpec, File)
import os
import re

class DWIMapsMatlabInputSpec(BaseInterfaceInputSpec):
    dwi_file = File(exists=True, mandatory=True)
    bvals_file = File(exists=True, mandatory=True)
    bvecs_file = File(exists=True, mandatory=True)
    b0indices = traits.Str(desc='b=0 induces', mandatory=True)
    omitindices = traits.Str(desc='omit induces', mandatory=True)
    dwimode = traits.Float(desc='Processing Mode', mandatory=True)
    script_file = File(exists=True, mandatory=True)
    matlabpath = File(exists=False, mandatory=True)
    localmatlabpath = File(exists=False, mandatory=False)


""" Ideally should be using this syntax from the nipype interfaces instead for source/template file naming.
        argstr="%s",
        position=1,
        name_source="in_files",
        name_template="%s_merged",
        hash_files=False,
    """

class DWIMapsMatlabOutputSpec(TraitedSpec):
    fadc_file = File(desc='output file fADC map')
    md_file = File(desc='output file MD map')
    fa_file = File(desc='output file FA map')
    meanrad_file = File(desc='output file MeanRad image')
    meanb0_file = File(desc='output file MeanB0 image')
    #runscript_file = File(desc='runable script')

class DWIMapsMatlab(BaseInterface):
    input_spec = DWIMapsMatlabInputSpec
    output_spec = DWIMapsMatlabOutputSpec

    def _run_interface(self, runtime):

        scriptname = os.path.basename(self.inputs.script_file)
        localscriptfile = os.path.join(os.path.abspath('.'),scriptname)

        # Load script
        with open(self.inputs.script_file) as script_fileid:
            script_content = script_fileid.read()

        script_content = script_content.replace('DUMMYDWIFILEIN', self.inputs.dwi_file)
        script_content = script_content.replace('DUMMYBVALSFILEIN', self.inputs.bvals_file)
        script_content = script_content.replace('DUMMYBVECSFILEIN', self.inputs.bvecs_file)
        script_content = script_content.replace('DUMMYBZEROINDICES', str(self.inputs.b0indices))
        script_content = script_content.replace('DUMMYOMITINDICES', str(self.inputs.omitindices))
        script_content = script_content.replace('DUMMYMODE', str(self.inputs.dwimode))
        script_content = script_content.replace('DUMMYPROJECTMATLABPATH', str(self.inputs.localmatlabpath))



        # Replace the input_image.mat file for the actual input of this interface
        with open(localscriptfile, 'w') as script_fid:
            script_fid.write(script_content)

        # Run a matlab command
        mlab = CommandLine(self.inputs.matlabpath + '/bin/matlab -nodesktop -nosplash -r ', args='\"run' + '(\'' + localscriptfile + '\')\"', terminal_output='stream')
        print(mlab)
        result = mlab.run()

        return result.runtime


    def _list_outputs(self):
        outputs = self._outputs().get()

        #THIS is critical!! use current (processing) path instead of asl_file path
        scriptname = os.path.basename(self.inputs.script_file)
        localscriptfile = os.path.join(os.path.abspath('.'),scriptname)
        curdir = os.path.dirname(localscriptfile)
        #output is in processing directory, but asl_file is in orig directory
        outputs['fadc_file'] = os.path.join(curdir,os.path.basename(self.inputs.dwi_file).replace('.nii','_Daxial.nii').replace('.gz',''))
        outputs['md_file'] = os.path.join(curdir,os.path.basename(self.inputs.dwi_file).replace('.nii','_MD.nii').replace('.gz',''))
        outputs['fa_file'] = os.path.join(curdir,os.path.basename(self.inputs.dwi_file).replace('.nii','_FA.nii').replace('.gz',''))
        outputs['meanrad_file'] = os.path.join(curdir,os.path.basename(self.inputs.dwi_file).replace('.nii','_MeanRad.nii').replace('.gz',''))
        outputs['meanb0_file'] = os.path.join(curdir,os.path.basename(self.inputs.dwi_file).replace('.nii','_MeanB0.nii').replace('.gz',''))
        #outputs['runscript_file'] = getattr(self, '_runscript_file')
        return outputs
