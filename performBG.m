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

function label = performBG(baseClsSegs, ECI, clsNum)

weightedB = bsxfun(@times, baseClsSegs, ECI); clear baseClsSegs
label = Tcut(weightedB',clsNum); 
