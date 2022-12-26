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

function [group, eigengap] = performSC(W, clsNum)

warning off

% calculate degree matrix
degs = sum(W, 2);
D    = sparse(1:size(W, 1), 1:size(W, 2), degs);

% compute unnormalized Laplacian
L = D - W;
k = max(clsNum);
% compute normalized Laplacian if needed

% avoid dividing by zero
degs(degs == 0) = eps;
% calculate D^(-1/2)
D = spdiags(1./(degs.^0.5), 0, size(D, 1), size(D, 2));
% calculate normalized Laplacian
L = D * L * D;

% compute the eigenvectors corresponding to the k smallest
% eigenvalues
[U, eigenvalue] = eigs(L, k, eps);
[a,b] = sort(diag(eigenvalue),'ascend');
eigenvalue = eigenvalue(:,b);
U = U(:,b);
eigengap = abs(diff(diag(eigenvalue)));
U = U(:,1:k);
U = real(U);
% in case of the Jordan-Weiss algorithm, we need to normalize
% the eigenvectors row-wise
% U = bsxfun(@rdivide, U, sqrt(sum(U.^2, 2)));
% U = U./repmat(sqrt(sum(U.^2,2)),1,size(U,2));

flag =0;
for ck = clsNum
    Cindex = find(clsNum==ck);
    UU = U(:,1:ck);
    UU = UU./repmat(sqrt(sum(UU.^2,2)),1,size(UU,2));
    temp = kmeans(UU,ck,'MaxIter',100,'Replicates',3);
    
    Cluster{Cindex} = temp;
end

if length(clsNum)==1
    group=Cluster{1};
else
    group = Cluster;
end
end
