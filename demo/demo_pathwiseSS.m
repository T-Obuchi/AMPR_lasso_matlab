%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Demonstration of pathwise evaluation of stability path (positive probability) 
% using AMPR in simulated dataset  
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
beta0(1:K0)=sqrt(sigmaB2)*randn(K0,1);      % Non-zero components of true signal
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
t1=toc

%% Plot of stability path
figure;
hold on;
for i=1:100:K0
plot(pathfit.lambda,pathfit.Pi(i,:),'b*-');
end
for i=K0+1:200:N
plot(pathfit.lambda,pathfit.Pi(i,:),'r*-');
end
xlabel('\lambda');
ylabel('\Pi');
set(gca,'XScale','Log');
title(['Some stability paths'],'Interpreter','latex');

%% Plot of confidence interval 
TP=pathfit.Pi(1:K0,:);    % Stability path for non-zero components
FP=pathfit.Pi(K0+1:N,:);  % Stability path for zero components

TP_med=median(TP);
TP_med_pos=quantile(TP,0.84)-TP_med;
TP_med_neg=TP_med-quantile(TP,0.16);
FP_med=median(FP);
FP_med_neg=FP_med-quantile(FP,0.16);
FP_med_pos=quantile(FP,0.84)-FP_med;

figure;
hold on;
e1=errorbar(lambdaV,FP_med,FP_med_neg,FP_med_pos,'ro');
e2=errorbar(lambdaV,TP_med,TP_med_neg,TP_med_pos,'bo');
set(gca,'XScale','log');
legend([e1 e2],{'FP','TP'},'Location','Best','interpreter','latex','FontSize',22);
ylim([0 1]);
xlabel('\lambda','FontSize',22);
ylabel('Probability','FontSize',22);


