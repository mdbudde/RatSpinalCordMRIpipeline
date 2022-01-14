function [Out, Imgs] = EstimateDTI(B0indices, Mode)
%
%
% inputs:
%   ImageFile - nifti file of all DWI images, including b=0 images
%   BvalFile - file specifying bvalues, 1xN text file
%   BvecFile - file specifying bvectors, 3xN text file
%   B0indices - vector of indices to b=0 images ([1:3] for example)
%               These can be true b=0 or others, depending on the mode
%               For example, a 'filtered' DW image in the case of
%               subsequent filter+probe DW images
%   Mode - diffusion scheme 0 = DTI; 1 = 2dDTI; 2 = single axis
%   Omitindices - Indices to remove from the dataset ([1] for example if
%                   using a filter-probe scheme.  Defaults to empty.
%                   Note that omit and B0 indices should not overlap, but
%                   if they do, omit takes presidence.
%
%
% outputs:
%   Out - structure of setup data and inputs
%   Imgs - structure of images calculated from the algorithm
%
%
%  MDBudde, SeungYi Lee, MCW, 3/2020
%


    %Dfreelim = 3; %unused for now, may be used with more advanced models.

    MatlabLocalPath = 'DUMMYPROJECTMATLABPATH';
    addpath(genpath(MatlabLocalPath))


    %values get replaced through nipype functions
    ImageFile = 'DUMMYDWIFILEIN';
    BvalFile = 'DUMMYBVALSFILEIN';
    BvecFile = 'DUMMYBVECSFILEIN';
    B0indices = DUMMYBZEROINDICES;
    Omitindices = DUMMYOMITINDICES;
    Mode = DUMMYMODE;

    Out = struct('filename',ImageFile);

    %Initial check for files
    if ~exist(BvalFile,'file')
        error('File %s does not exist or not reconstructed',BvalFile);
    end
    bvals = dlmread(BvalFile);

    if ~exist(BvecFile,'file')
        error('File %s does not exist or not reconstructed',BvecFile);
    end
    bvecs = dlmread(BvecFile);

    if ~exist(ImageFile,'file')
        error('File %s does not exist or not reconstructed',ImageFile);
    end


    fprintf('Reading dwi file...%s\n',ImageFile)
    images = load_untouch_nii(ImageFile);
    Out.dim = images.hdr.dime.dim(2:images.hdr.dime.dim(1));
    Out.pixdim = images.hdr.dime.pixdim(2:images.hdr.dime.dim(1));
    images.img = double(images.img);


    if ~exist('Omitindices','var')
        Omitindices = [];
    end

    %get initial dimensions
    nBvec = length(bvecs);
    nBval = length(bvals);
    nImg = size(images.img,4);

    %error checking for sizes and overlapping B0,omit indices
    allDims = [nBvec nBval nImg];
    nDirs = unique(allDims);
    if length(unique(allDims)) > 1
        error('Mismatch in lengths of bvalues, bvectors, or images');
    end
    if any(B0indices>nDirs)
        error('B0indices greater than number of dimensions');
    end
    if any(Omitindices>nDirs)
        error('Omit indices greater than number of dimensions');
    end

    intInd = intersect(B0indices,Omitindices);
    if any(intInd)
        error('B0 and Omit indices overlap.  Omit takes precidence');
    end


    %Omit indices are removed from vals, vecs, & images
    %b0 indices are updated accordingly for the reduced dataset
    initIndices = 1:nDirs;
    initIndices(Omitindices) = [];

    Out.Bvectors = bvecs(initIndices,:);
    Out.Bvalues = bvals(initIndices);
    Out.images = images.img(:,:,:,initIndices);

    curB0 = 1;
    newB0indices = zeros([1 length(B0indices)]);
    for ci = 1:length(initIndices)
        curind = initIndices(ci);
        if ismember(curind,B0indices)
            newB0indices(curB0) = ci;
            curB0 = curB0 + 1;
        end
    end

    if length(newB0indices>0) < 1
        error('No B0 remaining');
    end

    Out.B0indices = newB0indices;
    Out.DWindices = setxor(1:length(initIndices),newB0indices);


    % Check if Mode variable exists, default to DTI.
    if ~exist('Mode','var')
        Mode = 0;
    end

    %Setup data as needed for each type of fitting.
    if Mode == 1
        Out.Mode = 1;
        outputstring = '2dDTI';
        Out.Bvectors = normVecs(Out.Bvectors(:,1:2));
    elseif Mode == 2
        Out.Mode = 2;
        outputstring = '1Axis';
        Out.Bvectors = normVecs(Out.Bvectors(:,1));
    else
        Out.Mode = 0;
        outputstring = 'DTI';
        Out.Bvectors = normVecs(Out.Bvectors);
    end


    %setup output filename prefixe
    [fp, fn, fe] = fileparts(ImageFile);
    [fp, fn, fe] = fileparts(fn);
    OutputPrefix = fn; %strcat(fn,'_',outputstring);
    %OutputPrefix = fullfile(fp,OutputPrefix);

    %Run tensor or DWI Estimation calculations and create struct for saving maps
    Imgs = calcMultExpDiff(Out.images, Out.Bvalues, Out.Bvectors, Out.B0indices, Out.Mode);


    % This code gets each field from a structure and saves it
    % as a nifti file.  In this way, we don't have to specifically write out each map.  Just
    % include it as a field and makes saving more versitile under different fitting conditions.
    niftioutput = fieldnames(Imgs);
    for i=1:numel(niftioutput)
        niftisave(Imgs.(niftioutput{i}), OutputPrefix, niftioutput{i}, images);
        %Imgsout.(niftioutput{i}) = Imgs.(niftioutput{i});
    end

exit;
end %function


function Bvectors = normVecs(Bvectors)
    %normalize to unit vectors
    scaleFact = sqrt(sum(Bvectors.^2,2));
    scaleFact(scaleFact==0) = 1;
    Bvectors = Bvectors./repmat(scaleFact,[1 size(Bvectors,2)]);
end


% Calculate the DTI maps.
function [Dout] = calcMultExpDiff(imgs, bvals, bvecs, b0ind, Mode)
  imgsize = size(imgs);

  if size(bvals,1) == 1
      bvals = bvals';
  end

    switch Mode
        case 0

            W = zeros([length(bvecs) 7]); %7 for 6 unique tensor elements + a column of 1s
            %case 0 is the standard diffusion tensor [Dxx Dxy Dxz; Dxy Dyy Dyz; Dxz Dyz Dzz]
            for i=1:length(bvecs)
                W(i,2:7)=[bvecs(i,1)^2 bvecs(i,2)^2 bvecs(i,3)^2 2*bvecs(i,1)*bvecs(i,2) 2*bvecs(i,1)*bvecs(i,3) 2*bvecs(i,2)*bvecs(i,3)];
            end

            bgW = W.*repmat(-bvals/1000,[1 7]);
            bgW(:,1) = 1;
            Dindx=[2, 5, 6, 5, 3, 7, 6, 7, 4 ]; %elements of the diffusion tensor as in W above
            nD=3;
            xDevVec = [1 0 0];

        case 1
            %case 1 is a 2d tensor with in-plane estimation only, which is
            W = zeros([length(bvecs) 4]); %4 for 3 tensor elements + a column of 1s
            %case 0 is the standard diffusion tensor [Dxx Dxy; Dxy Dyy]
            for i=1:length(bvecs)
                W(i,2:4)=[bvecs(i,1)^2 bvecs(i,2)^2 2*bvecs(i,1)*bvecs(i,2) ];
            end

            bgW = W.*repmat(-bvals/1000,[1 4]);
            bgW(:,1) = 1;
            Dindx=[2, 4, 4, 3]; %elements of the diffusion tensor as in W above
            nD=2;
            xDevVec = [1 0];

        case 2
            %case 2 is a single axis along a single direction only
            W = zeros([length(bvecs) 2]); %4 for 3 tensor elements + a column of 1s
            %case 0 is the standard diffusion tensor [Dxx Dxy; Dxy Dyy]
            for i=1:length(bvecs)
                W(i,2)=[bvecs(i,1)^2];
            end

            bgW = W.*repmat(-bvals/1000,[1 2]);
            bgW(:,1) = 1;
            Dindx=[2]; %single element
            nD=1;
    end

    %these aren't used for now, but remain for more advanced fitting routines.
    fminopts = optimset('fminbnd');
    fminopts.MaxFunEvals = 100;
    fminopts.MaxIter = 100;
    fminopts.TolX = 1e-5;

    lsqopts = optimset('lsqnonneg');
    lsqopts.MaxFunEvals = 50;
    lsqopts.MaxIter = 50;
    lsqopts.TolX = 1e-3;

    Dout.Daxial = zeros(imgsize(1:3));
    Dout.MeanB0 = zeros(imgsize(1:3));
    if Mode ~= 2
        Dout.MD = zeros(imgsize(1:3));
        Dout.Dradial = zeros(imgsize(1:3));
        Dout.FA = zeros(imgsize(1:3));
        Dout.V1 = zeros([imgsize(1:3),nD]);
        Dout.xDeviation = zeros([imgsize(1:3)]);
    % else Mode == 2
    %      FiltFinalInd=[1:5,25:29];
    %      Dout.MeanRad = mean(imgs(:,:,:,FiltFinalInd),4);
    end


    fprintf('Processing...\n')
    %Voxel-wise looping
    for xx=1:imgsize(1)
    fprintf('%.0f %%\n',xx/imgsize(1)*100);
        for yy=1:imgsize(2)
          for zz=1:imgsize(3)

                %setup matrices for estimation
                Nsignal = squeeze(imgs(xx,yy,zz,:));
                meanb0signal = mean(Nsignal(b0ind));
                NsignalN = log(Nsignal./meanb0signal);

                %do the least squares estimation
                DD = real(bgW\NsignalN);

                %redo with weighting
                w = diag(exp(bgW*DD));
                DD=(bgW'*w*bgW)\(bgW'*w*NsignalN);

                %reshape the derived tensor to a square matrix
                D=DD(Dindx);
                D=reshape(D,[nD,nD]);

                %remove nan or inf values
                D(isnan(D)) = 0;
                D(isinf(D)) = 0;

                %perform eigen decomposition for eigenvalues/vectors
                [dv,de] = eig(D);
                de = diag(de);
                [de,ide] = sort(de,1,'descend');

                %Calculated fitdata from estimated diffusivity.
                ADC = mean(de);
                fitNsignalN = -bvals/1000*ADC;
                fitdata = fitNsignalN;
                ydata = NsignalN;

                % Calculate Adjusted Rsquared, from matlab help page: https://www.mathworks.com/help/matlab/data_analysis/linear-regression.html
                yresid = ydata - fitdata;
                SSresid = sum(yresid.^2);
                SStotal = (length(ydata)-1) * var(ydata);
                rsq= 1 - SSresid/SStotal;
%                rsq_adj = 1 - SSresid/SStotal * (length(ydata)-1)/(length(ydata)-length(gammVar));



                %output estimated values.
                Dout.Daxial(xx,yy,zz) = de(1);
                Dout.MeanB0(xx,yy,zz) = meanb0signal;
                Dout.Rsquared(xx,yy,zz,:) = rsq;
                if Mode ~= 2
                    Dout.MD(xx,yy,zz)=mean(de);
                    Dout.Dradial(xx,yy,zz) = mean(de(2:nD));
                    Dout.FA(xx,yy,zz)=sqrt(3/2)*sqrt(sum((de-mean(de)).^2))./sqrt(sum(de.^2));
                    Dout.V1(xx,yy,zz,:) = dv(:,ide(1));
                    if Mode == 1
                        angDev = atan2d(norm(cross2d(dv(:,ide(1)),xDevVec)),dot(dv(:,ide(1)),xDevVec));
                    else
                        angDev = atan2d(norm(cross(dv(:,ide(1)),xDevVec)),dot(dv(:,ide(1)),xDevVec));
                    end
                    if angDev > 90
                        angDev = (180 - angDev);
                    end
                    Dout.xDeviation(xx,yy,zz) = angDev;
                end

                if Mode == 2
                    Dout.V1(xx,yy,zz,:) = [1 0 0]; %dummy vector for later comparisons to centerline-estimation vectors.
                end
            end % yloop
        end % xloop
    end %z loop

    if Mode==2
        %Composite
         Dout.Composite = cat(5,Dout.MeanB0,Dout.Daxial);
    end

end


function niftisave(imagefile, Prefix, Suffix, niftifile)
    fname = strcat(Prefix,'_', Suffix,'.nii');   %Naming convention
%     fname = strcat(Prefix,'_', Suffix,'.nii');
    nii = niftifile;
    nii.img=imagefile;

    % these are essential for double valued output.  scaling factor is removed
    nii.hdr.dime.datatype = 64;
    nii.hdr.dime.bitpix = 64;
    nii.hdr.dime.scl_slope = 1;
    nii.hdr.dime.cal_max = 2;
    nii.hdr.dime.cal_min = 0;

    %The following lines edit the hdr for proper sizes
    %(otherwise errors occur using load_nii function)
    if size(imagefile,3)==1
        nii.hdr.dime.dim = [3 size(imagefile,1) size(imagefile,2) size(imagefile,3) 1 1 1 1];
        nii.original.hdr.dime.dim = [3 size(imagefile,1) size(imagefile,2) size(imagefile,3) 1 1 1 1];
    end
    if size(imagefile,4)==1
        nii.hdr.dime.dim = [3 size(imagefile,1) size(imagefile,2) size(imagefile,3) 1 1 1 1];
        nii.original.hdr.dime.dim = [3 size(imagefile,1) size(imagefile,2) size(imagefile,3) 1 1 1 1];
    end
    if size(imagefile,4)>1
        nii.hdr.dime.dim = [4 size(imagefile,1) size(imagefile,2) size(imagefile,3) size(imagefile,4) 1 1 1];
        nii.original.hdr.dime.dim = [4 size(imagefile,1) size(imagefile,2) size(imagefile,3) size(imagefile,4) 1 1 1];
    end
    if size(imagefile,5)>1
        nii.hdr.dime.dim = [5 size(imagefile,1) size(imagefile,2) size(imagefile,3) size(imagefile,4) size(imagefile,5) 1 1];
        nii.original.hdr.dime.dim = [5 size(imagefile,1) size(imagefile,2) size(imagefile,3)...
            size(imagefile,4) size(imagefile,5) 1 1];
    end
    fprintf('Writing %s\n',fname);
    save_untouch_nii(nii,fname);
end


function c = cross2d(a,b)
    c = [ a(1).*b(2)-a(2).*b(1)];
end
