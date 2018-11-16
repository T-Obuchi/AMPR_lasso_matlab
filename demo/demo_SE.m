%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Demonstration of State Evolution (SE) for 
% Approximate Message Passing with Resampling (AMPR) in simulated dataset 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% By Tomoyuki Obuchi
% Origial version was written on 2018 Nov. 13.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Method: 
%  See arXiv:1802.10254.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;

% Path 
addpath('../routine');

% Parameters 
alpha=0.5;            % Ratio of dataset size to model dimensionaltiy
N=20000;              % Model dimensionality (number of covariates)
M=ceil(alpha*N);      % Dataset size (number of measurements)
rho0=0.2;             % Ratio of non-zero components in synthetic data
K0=ceil(rho0*N);      % Number of non-zero components
sigmaN2=0.01;         % Component-wise noise strength 
sigmaB2=1./rho0;      % Component-wise signal strength

% Sample generation
seed=1;
rng(seed);
beta0=zeros(N,1);    
beta0(1:K0)=sqrt(sigmaB2)*randn(K0,1); % True signal
X=randn(M,N)/sqrt(N);                  % Covariates
Y=X*beta0+sqrt(sigmaN2)*randn(M,1);    % Responses

% Other parameters
lambda=1;        % l1 regularization coefficient 
w     =1;        % 1: no penalty randomization, 1/2: recommended in stability selection
p_w   =0;        % 0: no penalty randomization, 1/2: recommended in stability selection
tau   =1;        % 1: standard bootstrap,       1/2: recommended in stability selection

% AMPR 
chi_in=zeros(N,1);
W_in=zeros(N,1);
beta_in=zeros(N,1);
tic;
fit_AMPR=AMPR_lasso_track(Y,X,lambda,w,p_w,tau,beta_in,chi_in,W_in);
toc

% SE
chi_til_in=mean(chi_in);
W_til_in=mean(W_in);
MSE_in=norm(beta0-beta_in)^2/N;
tic;
fit_SE=SE_AMPR(alpha,sigmaN2,rho0,sigmaB2,lambda,w,p_w,tau,chi_til_in,W_til_in,MSE_in);
toc

% AMPR result 
MAXIT=size(fit_AMPR.beta,2);
chiV=zeros(MAXIT,1);
WV=zeros(MAXIT,1);
MSEV=zeros(MAXIT,1);
for i=1:MAXIT
    chiV(i)=mean(fit_AMPR.chi(:,i));
    WV(i)=mean(fit_AMPR.W(:,i));
    MSEV(i)=norm(beta0-fit_AMPR.beta(:,i))^2/N;
end
STEPS=[1:MAXIT]-1;

% SE result
MAXIT_SE=size(fit_SE.chi,1);
STEPS_SE=[1:MAXIT_SE]-1;

%% Comparison
hf=figure;
hold on;
hp=plot(STEPS,chiV,'b*',STEPS,WV,'go',STEPS,MSEV,'r+');
hp_SE=plot(STEPS_SE,fit_SE.chi,'b-',STEPS_SE,fit_SE.W,'g-',STEPS_SE,fit_SE.MSE,'r-');
xlabel('Iteration step t');
lgd=legend('$$\tilde{\chi}$$ (AMPR)','$$\tilde{W}$$ (AMPR)','MSE (AMPR)',...
    '$$\tilde{\chi}$$ (SE)','$$\tilde{W}$$ (SE)','MSE (SE)','Location','Best');
lgd.Interpreter='latex';
title(['$$\lambda=',num2str(lambda),',w=',num2str(w),',p_w=',num2str(p_w),',\tau=$$',num2str(tau)],'Interpreter','latex')
