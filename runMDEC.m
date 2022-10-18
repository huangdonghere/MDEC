%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is a demo for the MDEC-HC, MDEC-SC, and MDEC-BG algorithms,  %
% which are proposed in the following paper:                        %
%                                                                   %
% D. Huang, C.-D. Wang, J.-H. Lai, and C.-K. Kwoh.                  %
% "Toward Multi-Diversified Ensemble Clustering of High-Dimensional %
% Data: From Subspaces to Metrics and Beyond".                      %
% IEEE Transactions on Cybernetics, 2022, 52(11), pp.12231-12244.   %
% DOI: https://doi.org/10.1109/TCYB.2021.3049633                    %
%                                                                   %
% The code has been tested in Matlab R2016a and Matlab R2016b.      %
% GigHub: https://github.com/huangdonghere/MDEC                     %
% Written by Huang Dong. (huangdonghere@gmail.com)                  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Label_hc, Label_sc, Label_bg] = runMDEC(fea, K, M, para_tau,lowKNN,upKNN)

if nargin < 6
    upKNN = 20;
end
if nargin < 5
    lowKNN = 5;
end
if nargin < 4
    para_tau = 0.5;
end
if nargin < 3
    M = 30;
end

%% Generate base clusterins
disp(['.']);
disp(['Start generating ', num2str(M), ' base clusterings ... ']);
disp(['.']);
IDX = generateBaseCls(fea, M, para_tau, lowKNN, upKNN);
disp(['.']);
disp(['Ensemble generation completed.']);

disp(['.']);
disp(['Start consensus function ... ']);
[bcs, baseClsSegs] = getAllSegs(IDX);
ECI = getECI(bcs, baseClsSegs, 1);
S = getLWCA(baseClsSegs,ECI,M);

Label_hc = performHC(S, K);
disp(['.']);
disp(['MDEC-HC completed.']);

Label_sc = performSC(S, K);
disp(['.']);
disp(['MDEC-SC completed.']);

Label_bg = performBG(baseClsSegs, ECI, K); 
disp(['.']);
disp(['MDEC-BG completed.']);
