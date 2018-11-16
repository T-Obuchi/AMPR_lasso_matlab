function [fit]=SE_AMPR(alpha,sigmaN2,rho0,sigmaB2,lambda,w,p_w,tau,chi_in,W_in,MSE_in)
%--------------------------------------------------------------------------
% SE_AMPR.m: State evolution for AMPR.  
%--------------------------------------------------------------------------
%
% DESCRIPTION:
%    Solving state evolution (SE) equations for AMPR upto MAXIT (=30 in
%    default) steps.
%    The true signal's nonzero-component density is rho0, 
%    and the nonzero component is assumed to obey the zero-mean Gaussian
%    with variance sigmaB2.
%
% USAGE:
%    fit = SE_AMPR(alpha,sigmaN2,rho0,sigmaB2,lambda)
%    fit = SE_AMPR(alpha,sigmaN2,rho0,sigmaB2,lambda,w,p_w,tau)
%    fit = SE_AMPR(alpha,sigmaN2,rho0,sigmaB2,lambda,w,p_w,tau,chi_in,W_in,MSE_in)
%    (Use [] to apply the default value, e.g. 
%     fit = SE_AMPR(alpha,sigmaN2,rho0,sigmaB2,lambda,[],[],[],chi_in,W_in,MSE_in),
%     fit = SE_AMPR(alpha,sigmaN2,rho0,sigmaB2,lambda,w,p_w,tau,[],[],[]) )
% 
% INPUT ARGUMENTS:
%    alpha       Ratio of dataset size to model dimensionality
%
%    sigmaN2     Noise strength per component (assumed to be zero-mean Gaussian)
%
%    rho0        Non-zero component density of true signal
%
%    sigmaB2     Signal strength per non-zero components (assumed to be zero-mean Gaussian)
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
%    chi_in      Initial value of averaged intra-sample variance of covariates' coefficients. 
%                Not necessarily needed (start from zero if not specified).
%
%    W_in        Initial value of averaged inter-sample variance of covariates' coefficients. 
%                Not necessarily needed (start from zero if not specified).
%
%    MSE_in      Initial value of mean-sqaured error between the true and reconstructed signals. 
%                Not necessarily needed (start from rho0*sigmaB2 if not specified).
%
% OUTPUT ARGUMENTS:
%    fit         A structure.
%
%    fit.chi     Averaged intra-sample variance of covariates' coefficients. 
%
%    fit.W       Averaged inter-sample variance of covariates' coefficients. 
%
%    fit.MSE   Mean-sqaured error between the true and reconstructed signals.
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
%    computed by AMPR without numerical sampling over the randomness.
%    SE equations track the dynamical behavior of macroscopic quantities of AMPR. 
%
% REFERENCES:
%    [1] Tomoyuki Obuchi and Yoshiyuki Kabashima: Semi-analytic resampling in Lasso, 
%        arXiv:1802.10254.
%
% DEVELOPMENT:
%    12 Nov. 2018: Original version was written.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Parameters
if nargin < 5
    error('three input arguments needed at least');
end
if nargin < 6 || isempty(w) || w > 1 || w < 0
    w = 1;
end
if nargin < 6 || isempty(p_w) || p_w > 1 || p_w < 0
    p_w = 0;
end
if nargin < 6 || isempty(tau) || tau > 1 || tau < 0
    tau = 1;
end
if nargin < 9 || isempty(chi_in)
    chi_in = 0;
end
if nargin < 9 || isempty(W_in)
    W_in = 0;
end
if nargin < 9 || isempty(MSE_in)
    MSE_in = rho0*sigmaB2;
end

%%% Integration measures
% Poisson
CMAX=100;
c=[0:CMAX]';
Pc=poisspdf(c,tau);
Pc=Pc/sum(Pc);                 % Poisson measure

% Gaussian
MAX=10;
dz=.01;
z=[-MAX:dz:MAX];
u=[-MAX:dz:MAX]';
Nz=size(z);
Nu=size(u);
z_2d=repmat(z,Nu);
u_2d=repmat(u,Nz);
Dz=dz*exp(-z.^2/2)/sqrt(2*pi); 
Du=dz*exp(-u.^2/2)/sqrt(2*pi); 

% Lambda randomization
S_lam=lambda*[1/w,1];          % Set of lambda
P_lam=[p_w,1-p_w];             % Measure on set of lambda

% Running parameters and initial condition
MAXIT=30;
chi=chi_in;
W=W_in;
MSE=MSE_in;

% Save data
chiV=zeros(MAXIT,1);
WV=zeros(MAXIT,1);
MSEV=zeros(MAXIT,1);
chiV(1)=chi;
WV(1)=W;
MSEV(1)=MSE;

% main loop
for t=2:MAXIT
    
    % Main to conjugate
    chi_til=chi;
    W_til=W;
    f_in=c./(1+c*chi_til);
    f1=f_in'*Pc;
    f2=(f_in).^2'*Pc;
    
    % Second order parameters
    A=alpha*f1;
    C=alpha*f2*W_til+alpha*(f2-f1^2)*(MSEV(t-1)+sigmaN2);
    
    %%% Main parameters' update
    chi=0;
    W=0;
    MSE=0;

    % zero component's contribution
    v0=A^2*(MSEV(t-1)+sigmaN2)/alpha;
    chi=chi+( (1-rho0)/A )*( P_lam*( erfc( S_lam/sqrt(2*(v0+C))) )' );
    h=sqrt(v0)*u_2d+sqrt(C)*z_2d; 
    beta=soft_threshold_SE(A,h,S_lam);
    beta_ave  = ( P_lam(1)*beta{1}      + P_lam(2)*beta{2}      )*Dz';
    beta2_ave = ( P_lam(1)*(beta{1}.^2) + P_lam(2)*(beta{2}.^2) )*Dz';
    W=W+(1-rho0)*Du'*( beta2_ave-beta_ave.^2 );
    MSE=MSE+(1-rho0)*Du'*( beta_ave.^2 );
    
    % nonzero component's contribution
    v1=A^2*( sigmaB2 + (MSEV(t-1)+sigmaN2)/alpha );
    chi=chi+( rho0/A )*( P_lam*( erfc( S_lam/sqrt(2*(v1+C))) )' );    
    h_sx=sqrt(v1)*u_2d+sqrt(C)*z_2d;              % signal-crosstalk merged  
    beta=soft_threshold_SE(A,h_sx,S_lam);
    beta_ave_sx  = ( P_lam(1)*beta{1}      + P_lam(2)*beta{2}      )*Dz';
    beta2_ave_sx = ( P_lam(1)*(beta{1}.^2) + P_lam(2)*(beta{2}.^2) )*Dz';
    W=W+rho0*Du'*( beta2_ave_sx-beta_ave_sx.^2 );
    
    h_xb=A*sqrt(sigmaB2)*u_2d+sqrt(v0+C)*z_2d;    % crosstalk-bootstrap merged
    beta=soft_threshold_SE(A,h_xb,S_lam);
    beta_ave_xb  = ( P_lam(1)*beta{1}      + P_lam(2)*beta{2}      )*Dz';
    fst=sigmaB2;
    scd=-2*sqrt(sigmaB2)*(Du.*u)'*beta_ave_xb;
    trd=Du'*(beta_ave_sx.^2);
    MSE=MSE+rho0*(fst+scd+trd);
    
    % Save
    chiV(t)=chi;
    WV(t)=W;
    MSEV(t)=MSE;
end

fit.chi=chiV;
fit.W=WV;
fit.MSE=MSEV;

end