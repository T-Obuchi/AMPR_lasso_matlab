%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Comparison between AMPR and direct numerical resampling 
% in evaluation of the bootstrap average in simulated dataset 
%
% REQUIREMENT:
% For comparison, a direct numerical resampling is performed by using glmnet.
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

% Parameters for sample generation
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

%% AMPR 
tic;
fit_AMPR=AMPR_lasso(Y,X,lambda,w,p_w,tau);
t1=toc;

%% Numerical sampling using glmnet
NEXP=1000;
betaV=zeros(N,NEXP);
options=glmnetSet;
options.lambda=lambda/(M*tau);
options.intr=false;
options.standardize=false;
options.thresh =1.0e-10;
options.maxit =10^8;
tic;
for nexp=1:NEXP
    % Initialization
    rng(nexp);

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
    betaV(w_on,nexp)=w*fit.beta(w_on);
    betaV(w_off,nexp)=fit.beta(w_off);
end
t2=toc;

%%
disp([t1,t2]);  % elapsed time

% Mean value of beta
NEXP=size(betaV,2);
figure;
plot(fit_AMPR.beta,mean(betaV(:,1:NEXP),2),'bo',...
    [min(fit_AMPR.beta),max(fit_AMPR.beta)],...
    [min(fit_AMPR.beta),max(fit_AMPR.beta)],'k-');
title('$$ \overline{\beta}_i $$','Interpreter','latex');
xlabel('Semi-analitic');
ylabel('Numerical');
title(['$$\overline{\beta_i} (\lambda=',num2str(lambda),')$$'],'Interpreter','latex')

% Intra-sample variance
W_exp=sum(betaV(:,1:NEXP).^2,2)/NEXP-(sum(betaV(:,1:NEXP),2)/NEXP).^2;
figure;
plot(fit_AMPR.W,W_exp,'bo',...
    [min(W_exp),max(W_exp)],[min(W_exp),max(W_exp)],'k-');
xlabel('Semi-analitic');
ylabel('Numerical');
title(['$$ W_i (\lambda=',num2str(lambda),')$$'],'Interpreter','latex')

% Positive probability
thre=10^(-8);
MASK=abs(betaV)>thre;
P_pos=mean(MASK,2);
figure;
v_max=max(fit_AMPR.Pi);
v_min=min(fit_AMPR.Pi);
plot(fit_AMPR.Pi,P_pos,'bo',...
    [v_min,v_max],[v_min,v_max],'k-');
xlabel('Semi-analitic');
ylabel('Numerical');
title(['$$\Pi_i (\lambda=',num2str(lambda),')$$'],'Interpreter','latex')






