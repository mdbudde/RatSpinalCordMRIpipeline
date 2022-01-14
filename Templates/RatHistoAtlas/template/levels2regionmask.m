%function levels2regionmask(filein)

filein = 'RatHistoAtlas_levels.nii.gz';

addpath('~/github/BuddeDoDMRIprocessing/MatlabTools/Nifti');

        %C1:C7, T1:T13, L1:L6 
levels={[1:7], [8:20], [21:26]}; 
labels = {'cervical','thoracic','lumbar'};

nii = load_untouch_nii(filein)
niiout = nii;

for jj =1:length(levels)
    niiout.img = niiout.img.*0;
    for ll=1:length(levels{jj})
        niiout.img(nii.img==ll) = ll;
    end
    newfile = strrep(filein,'.nii',sprintf('_%s.nii',labels{jj}));
    save_untouch_nii(niiout,newfile);
end




