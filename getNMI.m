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

function allScores = getNMI(results,gt)

allScores = zeros(size(results,2),1);
for i = 1:size(results,2)
    if min(results(:,i))>0
        allScores(i) = NMImax(results(:,i), gt);
    end
end

function NMImax = NMImax(x, y)

assert(numel(x) == numel(y));
n = numel(x);
x = reshape(x,1,n);
y = reshape(y,1,n);

l = min(min(x),min(y));
x = x-l+1;
y = y-l+1;
k = max(max(x),max(y));

idx = 1:n;
Mx = sparse(idx,x,1,n,k,n);
My = sparse(idx,y,1,n,k,n);
Pxy = nonzeros(Mx'*My/n); %joint distribution of x and y
Hxy = -dot(Pxy,log(Pxy+eps));

Px = mean(Mx,1);
Py = mean(My,1);

% entropy of Py and Px
Hx = -dot(Px,log(Px+eps));
Hy = -dot(Py,log(Py+eps));

% mutual information
MI = Hx + Hy - Hxy;

% maximum normalized mutual information
NMImax = MI/max(Hx,Hy);
