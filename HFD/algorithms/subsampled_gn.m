function [params, options] = subsampled_gn(model,X, y,X_test, y_test, lambda,options)
%% subsampled Gauss-Newton-CG methods
% This mostly follows from Martens,2010: http://www.cs.toronto.edu/~jmartens/docs/HFDemo.zip
%
% input & output: 
%       model           ---- neural network model
%       X,y             ---- training data: input (d x n), output (c x n)
%       X_test,y_test   ---- test data
%       lambda          ---- l2 regularization
% .     options:
%           options.params:     weights of the model
%           options.tr_times:   training timestamp at each iteration
%           options.tr_losses:  training loss at each iteration
%           options.tr_grads:   training gradient norm at each iteration
%           opitons.tr_errs:    training error at each iteration
%           options.te_errs:    test error at each iteration
%           options.cur_ter:    current itertion number(if not 0, resume
%                               training
%           options.maxNoProps: maximum propagations for training
%           options.max_iters:  maximum iterations for training
%
%           options.name:       gauss-newton
%           options.eta1,eta2,gamma1,gamma2:
%                               parameters for adaptivity
%           options.hs:         subsampling Hessian size
%
%   

layersizes = model.layersizes;
numlayers = model.numlayers;
noProps = 1;
noMVPs = 1;
n = size(X,2);
sz = floor(0.05*n);
psize = layersizes(1,2:(numlayers+1))*layersizes(1,1:numlayers)' + sum(layersizes(2:(numlayers+1)));

delta = 5;
max_delta = 20;
eta1 = 0.75;
eta2 = 0.25;
gamma1 = 2/3;
gamma2 = 3/2;
theta = 10;
maxNoProps   = Inf;
maxMVPs = Inf;
max_iters = 100;
inner_iters = 100;
cur = 0;
cur_time = 0;
name = 'gauss-newton';

if isfield(options, 'delta')
    delta = options.delta;
end

if isfield(options, 'maxNoProps')
    maxNoProps  = options.maxNoProps;
end 

if isfield(options, 'maxMVPs')
    maxMVPs  = options.maxMVPs;
end 


if isfield(options, 'max_delta')
    max_delta = options.max_delta;
end

if isfield(options, 'eta1')
    eta1 = options.eta1;
    gamma1 = options.gamma1;
end

if isfield(options,'eta2')
    eta2 = options.eta2;
    gamma2 = options.gamma2;
end

if isfield(options,'hs')
    sz = options.hs;
end

if isfield(options,'max_iters')
    max_iters = options.max_iters;
end

if isfield(options,'inner_iters')
    inner_iters = options.inner_iters;
end

if isfield(options,'cur_iter') && options.cur_iter >= 1
    cur = options.cur_iter;
    cur_time = options.tr_times(cur);
    options.tr_times = [options.tr_times(1:cur); zeros(max_iters,1)];
    options.tr_losses = [options.tr_losses(1:cur); zeros(max_iters,1)];
    options.tr_grad = [options.tr_grad(1:cur); zeros(max_iters,1)];
    options.tr_errs = [options.tr_errs(1:cur); zeros(max_iters,1)];
    options.tr_noProps = [options.tr_noProps(1:cur); zeros(max_iters,1)];
    options.tr_noMVPs = [options.tr_noMVPs(1:cur); zeros(max_iters,1)];
    
    options.te_losses = [options.te_losses(1:cur); zeros(max_iters,1)];
    options.te_errs = [options.te_errs(1:cur); zeros(max_iters,1)];
    
    noProps = options.tr_noProps(cur);
    %maxNoProps  = maxNoProps  + noProps;
    noMVPs = options.tr_noMVPs(cur);
    maxMVPs = maxMVPs + noMVPs;
    
else
    options.tr_errs = zeros(max_iters,1);
    options.tr_losses = zeros(max_iters, 1);
    options.tr_grad = zeros(max_iters, 1);
    options.tr_times = zeros(max_iters, 1);
    options.tr_noProps = zeros(max_iters, 1);
    options.tr_noMVPs = zeros(max_iters, 1);
    
    options.te_errs = zeros(max_iters,1);
    options.te_losses = zeros(max_iters, 1);
end

if isfield(options,'name')
    name = options.name;
end
% initialize parameters
fprintf('initial setup:\n');
if isfield(options,'params')
    params = options.params;
else
    params = sprandn(psize,1,0.1)*0.5;
end
fprintf(' hession size: %d\n eta1: %g\n eta2: %g\n gamma1: %g\n gamma2: %g\n',...
    sz, eta1, eta2, gamma1, gamma2);
fprintf(' init delta: %g\n max delta : %g\n max iters for solver: %d\n max props: %g\n\n',...
    delta, max_delta, inner_iters, maxNoProps);
% load params.mat;
tic;
% training
fprintf('\n start training...\n');
s = zeros(psize, 1);
for iter = cur+1: cur + max_iters
    if noProps > maxNoProps || noMVPs > maxMVPs
        iter = iter - 1;
        break;
    end
    idx = randsample(n, sz);
    x_sample = X(:,idx);
    y_sample = y(:,idx);
    [~, ~, gauss_newton,~] = compute_gauss_newton(model, params, x_sample, y_sample);
%     [ll, grad, tr_err] = compute_gradient(model, params, X,y);
    [ll_err, grad] = compute_model(model, params, X,y);
    ll = ll_err(1); tr_err = ll_err(2);
    tr_loss = ll + 0.5 * lambda * (params'*params);
    grad = grad + lambda*params;
    GNV = @(V) gauss_newton(V)+lambda*V + theta*V;
    noProps = noProps + size(X,2);
%     [te_loss, te_err] = model_eval(model, params, X_test, y_test);
    te_loss_err = compute_model(model, params, X_test, y_test);
    te_loss = te_loss_err(1); te_err = te_loss_err(2);
    options.tr_losses(iter) = tr_loss;
    options.tr_errs(iter) = tr_err;
    options.te_losses(iter) = te_loss;
    options.te_errs(iter) = te_err;
    options.tr_grad(iter) = norm(grad,Inf);
    options.tr_times(iter) = toc + cur_time;
    options.tr_noProps(iter) = noProps;
    options.tr_noMVPs(iter) = noMVPs;
    fprintf('\nIter: %d, time = %g s\n', iter, options.tr_times(iter));
    fprintf('training loss + reg: %g, grad: %g(max), %g(norm)\n', tr_loss, norm(grad,Inf), norm(grad,2));
    fprintf('training err: %g\n', tr_err);
    fprintf('test loss: %g, test err: %g\n', te_loss, te_err);
    if norm(grad,Inf) <= 1E-16
        fprintf('Grad too small: %g\n',norm(grad,Inf));
        break;
    end
    
    % solve linear system
    Mdiag = grad.^2 + theta;
    [ss, is] = conjgrad_1(GNV, -grad, s, inner_iters, 5, Mdiag);
    noProps = noProps + is(end)*2*size(x_sample,2);
    noMVPs = noMVPs + is(end);
    %    [s,flag,relres,iters] = pcg(GNV, -grad,[],max_iters);
    s = ss{end};
    %    if flag ~= 0
    %        fprintf('flag: %d, CG did not converge.\n', flag);
    %    end
    % damping
    m = 1/2 * s'* GNV(s) + grad'*s;
    if m >= 0
        iter = iter -1;
        theta = theta * gamma2;
        continue;
    end
%     [newll,~] = model_eval(model,params + s, X,y);
    newll_err = compute_model(model, params + s, X,y);
    newll = newll_err(1); %new_err = newll_err(2);
    noProps = noProps + size(X,2);
    newll = newll + 0.5 * lambda * norm(params + s)^2;
    rho = (tr_loss - newll)/-m;
    if rho > eta1
        theta = theta * gamma1;
    elseif rho < eta2
        theta = theta * gamma2;
    end
    
    % line search
    rate = 1.0;
    c = 0.01;
    j = 0;
    while j < 60
        if newll <= tr_loss + c*rate*(grad'*s)
            break;
        else
            rate = rate * 0.8;
            j = j + 1;
        end
%         [newll, ~] = model_eval(model, params + rate*s, X, y);
        newll_err = compute_model(model, params + rate*s, X,y);
        newll = newll_err(1);
        noProps = noProps + size(X,2);
        newll = newll + 0.5 * lambda * norm(params + rate*s)^2;
    end
    if j == 60
        %completely reject the step
        j = Inf;
        rate = 0.0;
    end
    params = params + s*rate;
    fprintf('model reduction: %g, theta: %g, rho: %g, step-size: %g\niters: %d, total MVPs: %g, total Props: %g\n', m, theta, rho, rate, is(end), noMVPs, noProps);
end
options.params = params;
options.cur_iter = iter;
options.tr_times = options.tr_times(1:iter);
options.tr_losses = options.tr_losses(1:iter);
options.tr_errs = options.tr_errs(1:iter);
options.tr_grad = options.tr_grad(1:iter);
options.te_losses = options.te_losses(1:iter);
options.te_errs = options.te_errs(1:iter);
options.tr_noProps = options.tr_noProps(1:iter);
options.tr_noMVPs = options.tr_noMVPs(1:iter);
end
