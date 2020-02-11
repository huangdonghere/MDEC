%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                      %
% This is a demo for the three ensemble clustering algorithms (namely, % 
% MDEC-HC, MDEC-SC, and MDEC-BG) proposed in the paper below:          %                                                     %
%                                                                      %
% Dong Huang, Chang-Dong Wang, Jian-Huang Lai, and Chee-Keong Kwoh.    %
% "Toward Multi-Diversified Ensemble Clustering of High-Dimensional    %
% Data," under review, 2020.                                           %
%                                                                      %
% The code has been tested in Matlab R2016b and Matlab R2019b.         %
%                                                                      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function demo_1()
%% Run each algorithm multiple times and show its average performance.

clear all;
close all;
clc;


%% Load the data.
dataName = 'Yeoh02v1';
% dataName = 'MF';

% The variable 'fea' is the feature matrix where each row is a data sample.
% The variable 'gt' is the ground-truth label.
load(['data_',dataName,'.mat'],'fea','gt'); 

[N, D] = size(fea);

%% Set up
K = numel(unique(gt)); % The number of clusters
M = 100; % Ensemble size (i.e., the number of base clusterings)
para_tau = 0.5; % Sampling ratio
cntTimes = 10; % The number of times that each algorithm will be run.


%% Run MDEC
nmiScores = zeros(cntTimes,3);
for runIdx = 1:cntTimes
    disp('**************************************************************');
    disp(['Run ', num2str(runIdx),':']);
    disp('**************************************************************');
    
    % Perform the three algorithms under the MDEC framework.
    [result_MDEC_HC,result_MDEC_SC,result_MDEC_BG] = runMDEC3(fea, K, M, para_tau);  
    
    disp('--------------------------------------------------------------'); 
    nmiScores(runIdx,1) = getNMI(result_MDEC_HC,gt);
    nmiScores(runIdx,2) = getNMI(result_MDEC_SC,gt);
    nmiScores(runIdx,3) = getNMI(result_MDEC_BG,gt);
    disp(['The NMI score at Run ',num2str(runIdx)]);
    disp(['MDEC-HC: NMI = ',num2str(nmiScores(runIdx,1))]);
    disp(['MDEC-SC: NMI = ',num2str(nmiScores(runIdx,2))]);
    disp(['MDEC-BG: NMI = ',num2str(nmiScores(runIdx,3))]);
    disp('--------------------------------------------------------------');
end

disp('**************************************************************');
disp(['**** Ensemble clustering of the ',dataName,' dataset completed ****']);
disp(['Sample size: N = ', num2str(N)]);
disp(['Dimension:   D = ', num2str(D)]);
disp(['Ensemble size: M = ', num2str(M)]);
disp(['----------------- Average NMI over ',num2str(runIdx), ' runs ------------------']);
disp(['MDEC-HC: NMI = ',num2str(mean(nmiScores(:,1)))]);
disp(['MDEC-SC: NMI = ',num2str(mean(nmiScores(:,2)))]);
disp(['MDEC-BG: NMI = ',num2str(mean(nmiScores(:,3)))]);
disp('**************************************************************');