function [pathfit]=pathwiseSS_AMPR(Y,X,lambda,w,p_w,tau,gamma_min,gamma_max)
%--------------------------------------------------------------------------
% pathwiseSS_AMPR.m: Stability selection computed in a pathwise manner by 
% using Approximate Message Passing with Resampling (AMPR) 
%--------------------------------------------------------------------------
%
% DESCRIPTION:
%    Compute stability path (positive probability as a function of 
%    regularization parameter, see [1,2] for details) in a pathwise manner 
%    by using an approximate formula called AMPR.
%
% USAGE:
%    pathfit = pathwiseSS_AMPR(Y,X)
%    pathfit = pathwiseSS_AMPR(Y,X,lambda)
%    pathfit = pathwiseSS_AMPR(Y,X,lambda,w,p_w,tau)
%    (Use [] to apply the default value, e.g. 
%     pathfit = pathwiseSS_AMPR(Y,X,[],w,p_w,tau)  ) 
%
% INPUT ARGUMENTS:
%    Y           Response vector (M dimensional vector).
%
%    X           Matrix of covariates (M*N dimensional matrix).
%
%    lambda      l1 regularizaiton coefficients (L dimensional vector).   
%                If not specified, appropriate values are automatically
%                given.
%
%    w           Reweighting parameter to the regularization coefficients  
%                used in stability selection.
%                Default value is w=0.5 recommended for stability selection.
%
%    p_w         Fraction of randomization of the regularization coefficients 
%                used in stability selection. 
%                Default value is p_w=0.5 recommended for stability selection.
%
%    tau         Ratio of the size of bootstrap sample to the size of the original dataset.
%                Default value is tau=0.5 recommended for stability selection.
%
%    gamma_min   Minimum damping factor. (Not necessarily needed, default value is 1)
%
%    gamma_max   Maximum damping factor. (Not necessarily needed, default value is 1)
%
% OUTPUT ARGUMENTS:
%    pathfit         A structure. 
%
%    pathfit.beta    Mean value of covariates' coefficients (N*L dimensional vector). 
%
%    pathfit.W       Variances of covariates' coefficients (N*L dimensional vector). 
%
%    pathfit.Pi      Stability paths (N*L dimensional matrix).
%
%    pathfit.count   Iteration steps until convergence (L dimensional vector).
%
%    pathfit.lambda  l1 regularizaiton coefficients (L dimensional vector).
%
% DETAILS:
%    The present parametrization of Lasso is:
% 
%        \hat{beta}=argmin_{beta}
%            { (1/2)||Y-X*beta||_2^2 + \sum_{i}^{N}lambda_i * |beta_i| }
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
%    and the variance W and the positive probability Pi of each estimator 
%    are returned. See also AMPR_lasso.m for details.
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
%    24 Oct. 2018: Original version was written.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Parameters
[M,N]=size(X);
if nargin < 2
    error('three input arguments needed at least');
end
if nargin < 3 || isempty(lambda)
    Llam=100;
    lambda_max=round(max(abs(X'*Y)),1);
    lambda_min=10^(-2);
    rate=exp(log(lambda_min/lambda_max)/(Llam-1));
    lambda=lambda_max*(rate.^[0:Llam-1]);
else
    Llam=length(lambda);
end
if nargin < 4 || isempty(w) || w > 1 || w < 0
    w = 1/2;
end
if nargin < 5 || isempty(p_w) || p_w > 1 || p_w < 0
    p_w = 1/2;
end
if nargin < 6 || isempty(tau) || tau > 1 || tau < 0
    tau = 1/2;
end
if nargin < 7 || isempty(gamma_min)
    gamma_min=1;
end
if nargin < 8 || isempty(gamma_max)
    gamma_max=1;
end
if gamma_max < gamma_min
    gamma_max=gamma_min;
end
lambda=sort(lambda,'descend'); % Sort in descending order

% Initial condition
beta=zeros(N,1);
chi=zeros(N,1);
W=zeros(N,1);

% Pathwise evaluation of parameters using AMPR
pathfit.beta=zeros(N,Llam);
pathfit.W=zeros(N,Llam);
pathfit.Pi=zeros(N,Llam);
pathfit.count=zeros(N,Llam);
for ilam=1:Llam
    lambda_tmp=lambda(ilam);
    fit=AMPR_lasso(Y,X,lambda_tmp,w,p_w,tau,beta,chi,W,gamma_min,gamma_max);   
    pathfit.beta(:,ilam)=fit.beta;
    pathfit.W(:,ilam)=fit.W;
    pathfit.Pi(:,ilam)=fit.Pi;
    pathfit.count(:,ilam)=fit.count;
    beta=fit.beta;
    chi=fit.chi;
    W=fit.W;
%    [ilam lambda_tmp fit.count]
    if fit.flag==1
        warning(['AMPR did not converge and terminate at lambda=',num2str(lambda_tmp)...
            ,'. Smaller lambda is not examined']);
        break;
    end
end

% Output
pathfit.beta=pathfit.beta(:,1:ilam);
pathfit.W=pathfit.W(:,1:ilam);
pathfit.Pi=pathfit.Pi(:,1:ilam);
pathfit.count=pathfit.count(:,1:ilam);
pathfit.lambda=lambda(1:ilam);

end