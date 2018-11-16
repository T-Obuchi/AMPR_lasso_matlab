function [fit]=AMPR_lasso_track(Y,X,lambda,w,p_w,tau,beta_in,chi_in,W_in)
%--------------------------------------------------------------------------
% AMPR_lasso_track.m: Tracking first MAXIT (=30) steps of parameters of 
% Approximate message passing with resampling (AMPR). 
%--------------------------------------------------------------------------
%
% DESCRIPTION:
%    Tracking first MAXIT steps (default is 30 steps) of parameters of 
%    Approximate message passing with resampling (AMPR). 
%    No damping factor is applied.
%
% USAGE:
%    fit = AMPR_lasso_track(Y,X,lambda)
%    fit = AMPR_lasso_track(Y,X,lambda,w,p_w,tau)
%    fit = AMPR_lasso_track(Y,X,lambda,w,p_w,tau,beta_in,W_in,chi_in)
%    (Use [] to apply the default value, e.g. 
%     fit = AMPR_lasso_track(Y,X,lambda,[],[],[],beta_in,W_in,chi_in),
%     fit = AMPR_lasso_track(Y,X,lambda,w,p_w,tau,[],[],[])   )
% 
% INPUT ARGUMENTS:
%    Y           Response vector (M dimensional vector).
%
%    X           Matrix of covariates (M*N dimensional matrix).
%
%    lambda      l1 regularizaiton coefficient.   
%
%    w           Reweighting parameter to the regularization coefficients  
%                used in stability selection.
%                Default value is w=1 corresponding to the case of 
%                the non-randomized penalty.
%                A recommended value for stability selection is w=0.5.
%
%    p_w         Fraction of randomization of the regularization coefficients 
%                used in stability selection. 
%                Default value is p_w=0 corresponding to the case of 
%                the non-randomized penalty.
%                A recommended value for stability selection is p_w=0.5. 
%
%    tau         Ratio of the size of bootstrap sample to the size of the original dataset.
%                Default value is tau=1 corresponding to 
%                the Bootstrap method's convention.
%                A recommended value for stability selection is tau=0.5. 
%
%    beta_in     Initial estimate of mean value of covariates' coefficients (N dimensional vector). 
%                Not necessarily needed (but better to be appropriately given 
%                for faster convergence).
%
%    chi_in      Initial estimate of rescaled intra-sample variance 
%                of covariates' coefficients (N dimensional vector). 
%                Not necessarily needed (but better to be appropriately given 
%                for faster convergence).
%
%    W_in        Initial estimate of variance of covariates' coefficients (N dimensional vector). 
%                Not necessarily needed (but better to be appropriately given 
%                for faster convergence).
%
% OUTPUT ARGUMENTS:
%    fit         A structure.
%
%    fit.beta    Mean value of covariates' coefficients (N*MAXIT matrix). 
%
%    fit.chi     Rescaled intra-sample variance of covariates' coefficients (N*MAXIT matrix). 
%
%    fit.W       Variance of covariates' coefficients (N*MAXIT matrix). 
%
%    fit.Pi      Positive probabilities of covariates' coefficients (N*MAXIT matrix). 
%                (Probabilities such that covariates' coefficients take non-zero values.) 
%
%    fit.A, fit.B, fit.C
%                Parameters (N*MAXIT matrix) characterizing the 
%                probability distributions of covariates' coeffcients (see [1] for details).
%
% DETAILS:
%    Lasso is formulated as follows:
% 
%        \hat{beta}=argmin_{beta}
%            { (1/2)||Y-X*beta||_2^2 + \sum_{i}^{N}lambda_i*|beta_i| }
%
%    We consider the distribution of the estimator P(\hat{beta})  
%    when the bootstrap resampling of the dataset {X,Y} 
%    and the randomization to the penalty coefficients {lambda_i}_i are conducted. 
%    The penalty coefficient randomization is identically independently 
%    conducted through the following distribution (see [2] for details):
% 
%      P(lambda_i)=p_w*delta(lambda_i-lambda/w) + (1-p_w)*delta(lambda_i-lambda).
%
%    Here, the distribution of the estimator P(\hat{beta}) is approximately
%    computed by AMPR without numerical sampling over the randomness, 
%    and the first and second moments are returned, 
%    which are corresponding to beta, chi, and W in the output arguments.
%    The positive probability values P_pos are also returned. 
%    The other output arguments A,B,C are needed if the whole probability 
%    distribution is desired to be reconstructed.
%
% REFERENCES:
%    [1] Tomoyuki Obuchi and Yoshiyuki Kabashima: Semi-analytic resampling in Lasso, 
%        arXiv:1802.10254.
%
%    [2] Nicolai Meinshausen and Peter Buhlmann: Stability selection,
%        Journal of the Royal Statistical Society: Series B (Statistical
%        Methodology), 72(4):417--473, 2010.
%
% DEVELOPMENT:
%    13 Nov. 2018: Original version was written.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Parameters
X2=X.^2;
[M,N]=size(X);
if nargin < 3
    error('three input arguments needed at least');
end
if nargin < 4 || isempty(w) || w > 1 || w < 0
    w = 1;
end
if nargin < 4 || isempty(p_w) || p_w > 1 || p_w < 0
    p_w = 0;
end
if nargin < 4 || isempty(tau) || tau > 1 || tau < 0
    tau = 1;
end
if nargin < 7 || isempty(beta_in)
    beta_in = zeros(N,1);
end
if nargin < 7 || isempty(chi_in)
    chi_in = zeros(N,1);
end
if nargin < 7 || isempty(W_in)
    W_in = zeros(N,1);
end

% Integration Measures
CMAX=100;
c=[0:CMAX]';
Pc=poisspdf(c,tau);
Pc=Pc/sum(Pc);                 % Poisson measure
MAX=10;
dz=.01;
z=[-MAX:dz:MAX]';
Dz=dz*exp(-z.^2/2)/sqrt(2*pi); % Gaussian measure
S_lam=lambda*[1/w,1];          % Set of lambda
P_lam=[p_w,1-p_w];             % Measure on set of lambda

% Initial condition
f0=zeros(M,1);
f1=zeros(M,1);
f2=zeros(M,1);
chi_mu=X2*chi_in;
W_mu=X2*W_in;
for mu=1:M
    f0(mu)=Pc'*(  1./(1+c*chi_mu(mu))     );
    f1(mu)=Pc'*(  c./(1+c*chi_mu(mu))     );
    f2(mu)=Pc'*( (c./(1+c*chi_mu(mu))).^2 );        
end
beta=beta_in;
W=W_in;
chi=chi_in;
a=(f1.*(Y-X*beta))./f0;

% Output 
MAXIT=30;
fit.beta=zeros(N,MAXIT);
fit.chi=zeros(N,MAXIT);
fit.W=zeros(N,MAXIT);
fit.Pi=zeros(N,MAXIT);
fit.A=zeros(N,MAXIT);
fit.B=zeros(N,MAXIT);
fit.C=zeros(N,MAXIT);

% Initial condition
fit.beta(:,1)=beta_in;
fit.chi(:,1)=chi_in;
fit.W(:,1)=W_in;

% AMPR main loop
ERR=100;
gamma=1;    % gamma=1: No damping
for count=2:MAXIT
    beta_pre=beta;
    W_pre=W;
    
    % Moments to Conjugates 
    chi_mu=X2*chi;
    W_mu=X2*W;
    for mu=1:M
        f1(mu)=Pc'*(  c./(1+c*chi_mu(mu))     );
        f2(mu)=Pc'*( (c./(1+c*chi_mu(mu))).^2 );        
    end
    a=f1.*(Y-X*beta+chi_mu.*a);
    A=X2'*f1;
    B=X'*a+A.*beta;
    C=X2'*(  W_mu.*f2+(f2-f1.^2).*(a./f1).^2 );

    % Conjugates to Moments  
    for i=1:N
        b_tmp=soft_threshold_AMPR(A(i),B(i)+sqrt(C(i))*z,S_lam);
        
        beta(i)=(1-gamma)*beta(i)+gamma*(P_lam*(Dz'*b_tmp)');
        W(i)=(1-gamma)*W(i)+gamma*(P_lam*(Dz'*(b_tmp.^2))'-beta(i)^2);
        chi(i)=(1-gamma)*chi(i)+gamma*( 1/(2*A(i)) )...
            *( P_lam*(erfc((S_lam-B(i))/sqrt(2*C(i))))' + P_lam*(erfc((S_lam+B(i))/sqrt(2*C(i))))' );
    end
    
    % Positive probabilities
    P_pos=zeros(N,1);
    for i=1:N
        P_pos(i)=0.5*( P_lam*(erfc((S_lam-B(i))/sqrt(2*C(i)))+erfc((S_lam+B(i))/sqrt(2*C(i))))' );
    end
    
    % Error monitoring
    ERR=norm(beta-beta_pre)/norm(beta)+norm(W-W_pre)/norm(W);

    % Save
    fit.beta(:,count)=beta;
    fit.chi(:,count)=chi;
    fit.W(:,count)=W;
    fit.Pi(:,count)=P_pos;
    fit.A(:,count)=A;
    fit.B(:,count)=B;
    fit.C(:,count)=C;
    
end

end