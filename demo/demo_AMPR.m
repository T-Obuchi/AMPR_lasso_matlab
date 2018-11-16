%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Demonstration of Approximate Message Passing with Resampling (AMPR)
% in simulated dataset 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% By Tomoyuki Obuchi
% Origial version was written on 2018 Oct. 25.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Method: 
%  See arXiv:1802.10254.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;

% Path 
addpath('../routine');

% Parameters 
alpha=0.5;             % Ratio of dataset size to model dimensionaltiy
N=1000;                % Model dimensionality (number of covariates)
M=ceil(alpha*N);       % Dataset size (number of responses)
rho0=0.2;              % Ratio of non-zero components in synthetic data
K0=ceil(rho0*N);       % Number of non-zero components
sigmaN2=0.01;          % Component-wise noise strength 
sigmaB2=1./rho0;       % Component-wise signal strength

% Sample generation
seed=1;
rng(seed);
beta0=zeros(N,1);    
beta0(1:K0)=sqrt(sigmaB2)*randn(K0,1); % True signal
X=randn(M,N)/sqrt(N);                  % Covariates
Y=X*beta0+sqrt(sigmaN2)*randn(M,1);    % Responses

% Other parameters
lambda=1;          % l1 regularization coefficient 
w     =0.5;        % 1: no penalty randomization, 0.5: recommended in stability selection
p_w   =0.5;        % 0: no penalty randomization, 0.5: recommended in stability selection
tau   =0.5;        % 1: standard bootstrap,       0.5: recommended in stability selection

% AMPR 
tic;
fit_AMPR=AMPR_lasso(Y,X,lambda,w,p_w,tau);
t1=toc;

%%
% beta vs W
figure;
subplot(3,1,1);
plot(fit_AMPR.beta,'b*');
title(['$$(\lambda=',num2str(lambda),')$$'],'Interpreter','latex')
ylabel('$$\overline{\beta}$$','Interpreter','latex');
subplot(3,1,2);
plot(fit_AMPR.W,'go');
ylabel('W','Interpreter','latex');
subplot(3,1,3);
plot(fit_AMPR.Pi,'r+');
ylabel('$$\Pi $$','Interpreter','latex');
xlabel('INDEX');



