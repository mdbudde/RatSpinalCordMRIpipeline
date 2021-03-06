function Fit = bSSFP_fit(MTdata, Prot, FitOpt)

% ----------------------------------------------------------------------------------------------------
% bSSFP_fit Fits analytical SPGR model to data
% ----------------------------------------------------------------------------------------------------
% MTdata = struct with fields 'MTdata', and optionnaly 'Mask','R1map','B1map','B0map'
% Output : Fit structure with fitted parameters
% ----------------------------------------------------------------------------------------------------
% Written by: Jean-Fran?ois Cabana, 2016
% ----------------------------------------------------------------------------------------------------
% If you use qMRLab in your work, please cite :

% Cabana, J.-F., Gu, Y., Boudreau, M., Levesque, I. R., Atchia, Y., Sled, J. G., Narayanan, S.,
% Arnold, D. L., Pike, G. B., Cohen-Adad, J., Duval, T., Vuong, M.-T. and Stikov, N. (2016),
% Quantitative magnetization transfer imaging made easy with qMTLab: Software for data simulation,
% analysis, and visualization. Concepts Magn. Reson.. doi: 10.1002/cmr.a.21357
% ----------------------------------------------------------------------------------------------------

[alpha, Trf, TR, W] = bSSFP_prepare(Prot, FitOpt);

% Apply B1map
if (isfield(FitOpt,'B1') && ~isempty(FitOpt.B1))
    alpha = alpha * FitOpt.B1;
end

xData = [alpha, Trf, TR, W];

% Use R1map
if (isfield(FitOpt,'R1') && ~isempty(FitOpt.R1) && FitOpt.R1map)
    FitOpt.fx(3) = 1;
    FitOpt.st(3) = FitOpt.R1;
end

% Fix R1r = R1f
if (FitOpt.R1reqR1f)
    FitOpt.fx(4) = 1;
    FitOpt.st(4) = FitOpt.st(3);
end
   
fix = FitOpt.fx;

% Fitting
opt.Display = 'off';

[x_free, resnorm, residuals] = lsqcurvefit(@(x,xdata) bSSFP_fun(choose( FitOpt.st, x, fix ), xdata, FitOpt),...
                     FitOpt.st(~fix), xData, MTdata, FitOpt.lb(~fix), FitOpt.ub(~fix), opt);
     
x = choose( FitOpt.st, x_free, fix );     

% Fit results
Fit.F   = x(1);
Fit.kr  = x(2);
Fit.R1f = x(3);
Fit.R1r = x(4);
Fit.T2f = x(5);
Fit.M0f = x(6);
Fit.kf   = Fit.kr .* Fit.F;
Fit.M0r  = Fit.F .* Fit.M0f;

if (FitOpt.R1reqR1f)
     Fit.R1r = Fit.R1f;
end

if (isfield(FitOpt,'R1') && ~isempty(FitOpt.R1) && FitOpt.R1map)
     Fit.R1f = computeR1(Fit, FitOpt.R1);
end

% Fit.residuals = residuals;
Fit.resnorm = resnorm;

function a = choose( a, x, fx )
    a(~fx) = x;
end

end
