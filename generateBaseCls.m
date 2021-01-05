%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is a demo for the MDEC-HC, MDEC-SC, and MDEC-BG algorithms,  %
% which are proposed in the following paper:                        %
%                                                                   %
% D. Huang, C.-D. Wang, J.-H. Lai, and C.-K. Kwoh.                  %
% "Toward Multi-Diversified Ensemble Clustering of High-Dimensional %
% Data: From Subspaces to Metrics and Beyond".                      %
% IEEE Transactions on Cybernetics, accepted, 2021.                 %
% DOI: https://doi.org/10.1109/TCYB.2021.3049633                    %
%                                                                   %
% The code has been tested in Matlab R2016a and Matlab R2016b.      %
% GigHub: https://github.com/huangdonghere/MDEC                     %
% Written by Huang Dong. (huangdonghere@gmail.com)                  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function members = generateBaseCls(data, M, sRatio, lowKNN, upKNN)

[N,D] = size(data);

rand('state',sum(100*clock)*rand(1));
knnKs = ceil(rand(M,1)*min(upKNN-lowKNN+1,N-lowKNN))+lowKNN-1;
rand('state',sum(100*clock)*rand(1));
alphas = rand(M,1)*0.6+0.2;


subspaces = [];
numD = round(sRatio*D);
for iSS = 1:M
%     subspaces{iSS} = selectSubspaceIdxsWrtSizesII(straL, sRatio);
    rand('state',sum(100*clock)*rand(1));
    subspaces{iSS} = randperm(D,numD);
end

members = ones(N, M);
rand('state',sum(100*clock)*rand(1));
Ks = ceil(rand(M,1)*(ceil(sqrt(N)-1))+1);
parfor iK = 1:M
    tic1 = tic;
    Dist1 = dist2(data(:,subspaces{iK}),data(:,subspaces{iK}));
    W1 = getSparseAffinityMatrix(Dist1, knnKs(iK), alphas(iK));
    try
        members(:,iK) = performSC(W1,Ks(iK));
    catch
    end
    toc(tic1);
end

function n2 = dist2(x, c)
%DIST2	Calculates squared distance between two sets of points.
%
%	Description
%	D = DIST2(X, C) takes two matrices of vectors and calculates the
%	squared Euclidean distance between them.  Both matrices must be of
%	the same column dimension.  If X has M rows and N columns, and C has
%	L rows and N columns, then the result has M rows and L columns.  The
%	I, Jth entry is the  squared distance from the Ith row of X to the
%	Jth row of C.
%
%	See also
%	GMMACTIV, KMEANS, RBFFWD
%

%	Copyright (c) Christopher M Bishop, Ian T Nabney (1996, 1997)

[ndata, dimx] = size(x);
[ncentres, dimc] = size(c);
if dimx ~= dimc
	error('Data dimension does not match dimension of centres')
end

n2 = (ones(ncentres, 1) * sum((x.^2)', 1))' + ...
  		ones(ndata, 1) * sum((c.^2)',1) - ...
  		2.*(x*(c'));

%%
function [W]=affinityMatrix(Diff,K,sigma);
% Computes an affinity matrix for a given distance matrix
if nargin<3
    sigma = 0.5;
end
if nargin<2
    K = 20;
end

Diff=(Diff+Diff')/2;
Diff = Diff - diag(diag(Diff));
[T,INDEX]=sort(Diff,2);
[m,n]=size(Diff);
W=zeros(m,n);
TT=mean(T(:,2:K+1),2)+eps;
Sig=(repmat(TT,1,n)+repmat(TT',n,1) + 1*Diff)/3;
Sig=Sig.*(Sig>eps)+eps;
W=normpdf(Diff,0,sigma*Sig);
W = (W + W')/2;
return