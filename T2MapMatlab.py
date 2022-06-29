from nipype.interfaces.base import (CommandLine, traits, TraitedSpec,
                                    BaseInterface, BaseInterfaceInputSpec, File)


"""
Nipype interface to use matlab to create T2 maps through matlab.

It effectively takes a default matlab script, and replaces inputs with explict datapaths.
Then it simply runs the script.  This is easier than passing function values based on my experience.


"""

import os
import re

class T2MapMatlabInputSpec(BaseInterfaceInputSpec):
    in_files = traits.List(File(exists=True), argstr="%s", mandatory=True)
    tetimes_file = File(exists=True, mandatory=True)
    script_file = File(exists=True, mandatory=True)
    matlabpath = File(exists=False, mandatory=True)
    localmatlabpath = File(exists=False, mandatory=False)

class T2MapMatlabOutputSpec(TraitedSpec):
    t2map_file = File(desc='output file T2 map')
    #runscript_file = File(desc='runable script')

class T2MapMatlab(BaseInterface):
    input_spec = T2MapMatlabInputSpec
    output_spec = T2MapMatlabOutputSpec

    def _run_interface(self, runtime):

        scriptname = os.path.basename(self.inputs.script_file)
        localscriptfile = os.path.join(os.path.abspath('.'),scriptname)

        # Load script
        with open(self.inputs.script_file) as script_fileid:
            script_content = script_fileid.read()

        infilescell = str(self.inputs.in_files)
        infilescell = infilescell.replace('[','{')
        infilescell = infilescell.replace(']','}')

        script_content = script_content.replace('DUMMYINFILEIN', infilescell)
        script_content = script_content.replace('DUMMYTITIMESFILEIN', self.inputs.tetimes_file)
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
        outputs['t2map_file'] = os.path.join(curdir,os.path.basename(self.inputs.in_files[0]).replace('.nii','_T2map.nii').replace('.gz',''))
        #outputs['runscript_file'] = getattr(self, '_runscript_file')
        return outputs
