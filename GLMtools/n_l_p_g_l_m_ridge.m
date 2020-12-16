function [neg_logli, grad, hess_mtx] = ...
    n_l_p_g_l_m_ridge(p, x, y, lambda, dt)
% [neglogli, dL, H] = Loss_GLM_logli_exp(prs,XX);
%
% Compute negative log-likelihood of data under a Poisson GLM model with
% ridge regularization.
%
% Inputs:
%     p : numeric array [1 X n_params]
%         The parameters' weights vector.
%     x : numeric array [n_obs X n_params]
%         The design matrix.
%     y : numeric array [n_obs X 1]
%         The response variable vector.
%     lambda : numeric array [1 X 1]
%         The ridge parameter.
%     dt : numeric array [1 X 1]
%         The time bin size used.
%
% Outputs:
%     n_l : numeric array [1 X 1] 
%         The negative log-likelihood of the spike train.
%     grad : numeric array [n_params X 1]
%         The gradient (the optimized params).
%     hess_mtx : numerica array [n_params X n_params]
%         The hessian matrix.

% Compute GLM filter output and condititional intensity.
y_raw = x * p;              % model's raw output
y_pred = exp(y_raw) * dt;   % model's predicted output (per bin)

% Compute neg log-likelihood.
neg_logli = -(y' * y_raw - sum(y_pred) - sum(log(factorial(y))));
neg_logli = neg_logli + lambda * sum(p .^ 2);  % add ridge penalty

neg_logli = -(y' * (x * p) - sum(exp(x * p) * dt) - sum(log(factorial(y))));

% dL1: -y' * x
% dL2: x' * exp(x * p) (d/dp(x * p) = x)

% Compute Gradient
if (nargout > 1)
    dL1 = -x' * y;
    dL0 = x' * y_pred;
    grad = dL1 + dL0;  % x' * (y_pred - y) -> error
    grad = grad + lambda * p;  % add ridge penalty?
end

% Compute Hessian
if nargout > 2
    hess_mtx = x'*bsxfun(@times,x,y_pred); % non-spiking term 
end
