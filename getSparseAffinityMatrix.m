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

function [W]=getSparseAffinityMatrix(Diff,K,sigma)

if nargin<3
    sigma = 0.5;
end
if nargin<2
    K = 20;
end

Diff=(Diff+Diff')/2;

o_Diff = Diff;
[nSmp,mSmp] = size(Diff);
dump = zeros(nSmp,K);
idx = dump;
for i = 1:min(nSmp,mSmp)
    Diff(i,i) = 1e100;
end
for i = 1:K
    [dump(:,i),idx(:,i)] = min(Diff,[],2);
    temp = (idx(:,i)-1)*nSmp+[1:nSmp]';
    Diff(temp) = 1e100; 
end
Diff = o_Diff; clear o_Diff
Diff = Diff - diag(diag(Diff));

[m,n]=size(Diff);
W=zeros(m,n);
TT=mean(dump,2)+eps;
Sig=(repmat(TT,1,n)+repmat(TT',n,1) + 1*Diff)/3;
Sig=Sig.*(Sig>eps)+eps;
W=normpdf(Diff,0,sigma*Sig);

W = (W + W')/2;

% Preserving KNN
%% Fasten by HD
dump = zeros(nSmp,K);
idx = dump;
W = W - diag(diag(W));
for i = 1:K
    [dump(:,i),idx(:,i)] = max(W,[],2);
    temp = (idx(:,i)-1)*nSmp+[1:nSmp]';
    W(temp) = 0; 
end
Gidx = repmat([1:nSmp]',1,K);
Gjdx = idx;
W=sparse(Gidx(:),Gjdx(:),dump,nSmp,mSmp);

W = max(W,W');
