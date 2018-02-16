function [params, options] = subsampled_tr_cg(model,X, y,X_test, y_test, lambda,options)
%% subsampled trust-region method for deep learning
% input & ouput: 
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

%           options.name:       trust-region
%           options.delta:      initial trust-region radius
%           options.eta1,eta2,gamma1,gamma2:
%                               parameters for adaptivity
%           options.hs:         subsampling Hessian size
%
% written by Peng Xu, Fred Roosta, 6/8/2017, updated(2/8/2018)

layersizes = model.layersizes;
numlayers = model.numlayers;
noProps = 1;
noMVPs = 1;
n = size(X,2);
sz = floor(0.05*n);
psize = layersizes(1,2:(numlayers+1))*layersizes(1,1:numlayers)' + sum(layersizes(2:(numlayers+1)));

delta = 5;
max_delta = 20;
eta1 = 0.8;
eta2 = 1E-4;
gamma1 = 2;
gamma2 = 1.2;
maxNoProps = Inf;
maxMVPs = Inf;
max_iters = 100;
inner_iters = 100;
cur = 0;
cur_time = 0;
name = 'trust-region';

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
for iter = cur+1: cur + max_iters
    if noProps > maxNoProps || noMVPs > maxMVPs
        iter = iter - 1;
        break;
    end
    idx = randsample(n, sz);
    x_sample = X(:,idx);
    y_sample = y(:,idx);
    [~, ~, hess,~] = compute_model(model, params, x_sample, y_sample);
    [ll_err, grad] = compute_model(model, params, X,y);
    ll = ll_err(1); tr_err = ll_err(2);
    tr_loss = ll + 0.5 * lambda * (params'*params);
    grad = grad + lambda*params;
    HessV = @(V) hess(V)+lambda*V;
    noProps = noProps + size(X,2);
    te_loss_err = compute_model(model, params, X_test, y_test);
    te_loss = te_loss_err(1); te_err = te_loss_err(2);
    
    options.tr_losses(iter) = tr_loss;
    options.tr_errs(iter) = tr_err;
    options.te_losses(iter) = te_loss;
    options.te_errs(iter) = te_err;
    options.tr_grad(iter) = norm(grad,Inf);
    options.tr_noProps(iter) = noProps;
    options.tr_noMVPs(iter) = noMVPs;
    options.tr_times(iter) = toc + cur_time;
    fprintf('\nIter: %d, time = %g s\n', iter, options.tr_times(iter));
    fprintf('training loss + reg: %g, grad: %g(max), %g(norm)\n', tr_loss, norm(grad,Inf), norm(grad,2));
    fprintf('training err: %g\n', tr_err);
    fprintf('test loss: %g, test err: %g\n', te_loss, te_err);
    if norm(grad,Inf) <= 1E-16
        fprintf('Grad too small: %g\n',norm(grad,Inf));
        break;
    end
%     if tr_err <= 1E-6
%         fprintf('Training err too samll: %g\n', tr_err);
%         break;
%     end
    
    % solve trust-region subproblem
    fail_count = 0;
    while true
        steihaugParams = [1e-9, 250, 0];
        if fail_count == 0
            s0 = randn(psize,1);
            s0 = 0.99*delta*s0/norm(s0);
        end
        [s,m, num_cg, iflag] = cg_steihaug(HessV, grad, delta, steihaugParams, s0 );
        fprintf('Steihaug solution: %s\n',iflag);
        noProps = noProps + num_cg*2*size(x_sample,2);
        noMVPs = noMVPs + num_cg;
        if m >= 0
            s = 0;
            break;
        end
        fprintf('model reduction: %g\n', m);
        % [newll,~] = model_eval(model,params + s, X,y);
        newll_err = compute_model(model, params + s, X,y);
        newll = newll_err(1); %new_err = newll_err(2);
        noProps = noProps + size(X,2);
        newll = newll + 0.5 * lambda * norm(params + s)^2;
        rho = (tr_loss - newll)/-m;
        if rho < eta2
            fail_count = fail_count + 1;
            fprintf('FALIURE No. %d: delta = %g, rho = %g, iters: %g\n', fail_count, delta, rho,num_cg);
            delta = delta/gamma1;
            s0 = delta*s/norm(s);
        elseif rho < eta1
            fprintf('SUCCESS: delta = %g, rho = %g, s = %g\niters: %g, total MVPs: %g, total Props: %g\n', delta, rho, norm(s), num_cg, noMVPs, noProps);
            params = params + s;
            delta = min(max_delta, gamma2*delta);
            break;
        else
            fprintf('SUPER SUCCESS: delta = %g, rho = %g, s = %g\niters: %g, total MVPs: %g, total Props: %g\n', delta, rho, norm(s),num_cg,noMVPs, noProps);
            params = params + s;
            delta = min(max_delta, gamma1*delta);
            break;
        end
    end
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