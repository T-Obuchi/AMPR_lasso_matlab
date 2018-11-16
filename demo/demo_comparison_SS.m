%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Comparison betweeen AMPR and direct numerical sampling 
% in pathwise evaluation of stability path (positive probability) 
%
% REQUIREMENT:
% For comparison, a direct numerical resampling is performed by using glmnet.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% By Tomoyuki Obuchi
% Origial version was written on 2018 Nov. 15.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Method: 
%  See arXiv:1802.10254.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;

% Path 
addpath('../routine');

% Parameters for sample generation
alpha=5;               % Ratio of dataset size to model dimensionaltiy
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
beta0(1:K0)=sqrt(sigmaB2)*randn(K0,1);      % True signal
X=randn(M,N)/sqrt(N);                       % Covariates
Y=X*beta0+sqrt(sigmaN2)*randn(M,1);         % Responses

% Other parameters
lambdaV=[3.00:-0.04:0.04];  % l1 coefficients
w      =0.5;                % 1: no penalty randomization, 0.5: recommended in stability selection
p_w    =0.5;                % 0: no penalty randomization, 0.5: recommended in stability selection
tau    =0.5;                % 1: standard bootstrap,       0.5: recommended in stability selection

%% AMPR 
tic;
[pathfit]=pathwiseSS_AMPR(Y,X,lambdaV,w,p_w,tau);
t1=toc;

%% Numerical sampling for stability selection using glmnet
NEXP=1000;
options=glmnetSet;
options.intr=false;
options.standardize=false;
options.thresh =1.0e-10;
options.maxit =10^8;
options.lambda=lambdaV/(M*tau);

thre=10^(-8);
Llam=length(lambdaV);
COUNT=zeros(N,Llam);

tic;
for nexp=1:NEXP
    % Initialization
    rng(nexp);
    nexp
    betaV=zeros(N,Llam);

    % Penalty coeff. randomization
    r1=rand(N,1);
    w_on=r1<p_w;
    w_off=not(w_on);

    % Data resampling    
    r2=rand(M*tau,1);
    Ibs=ceil(r2*M);
    Ybs=Y(Ibs);
    Xbs=X(Ibs,:);    
    M_tmp=size(Ybs,1);

    % Reweighting columns by coeff. randomization 
    Xmod=zeros(M_tmp,N);
    Xmod(:,w_on)=w*Xbs(:,w_on);
    Xmod(:,w_off)=Xbs(:,w_off);       

    % Glmnet
    fit=glmnet(Xmod,Ybs,'gaussian',options);

    % Recovering original weight
    betaV(w_on,:)=w*fit.beta(w_on,:);
    betaV(w_off,:)=fit.beta(w_off,:);

    % Counting non-zero components
    COUNT=COUNT+cast(abs(betaV)>thre,'double');
   
end
t2=toc;
Pi_exp=COUNT/NEXP; % Stability path

%% Plot of stability path
disp([t1,t2]);  % elapsed time
figure;
hold on;
for i=1:100:K0
plot(pathfit.lambda,pathfit.Pi(i,:),'ro',lambdaV,Pi_exp(i,:),'b*');
end
for i=K0+1:200:N
plot(pathfit.lambda,pathfit.Pi(i,:),'ro',lambdaV,Pi_exp(i,:),'b*');
end
xlabel('\lambda');
ylabel('\Pi');
set(gca,'XScale','Log');
title(['Some stability paths'],'Interpreter','latex')
legend('AMPR','Numerical','Location','Best');



