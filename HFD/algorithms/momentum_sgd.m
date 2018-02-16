function [params, options] = momentum_sgd(model,X, y,X_test, y_test, lambda,options)
%% Momentum SGD training on Neural Network Models.
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

%           options.name:       sgd
%           options.alpha:      step size
%           options.beta:       momentum parameter
%           options.hs:         subsampling gradient size
%
% written by Peng Xu, Fred Roosta, 6/8/2017, updated(2/8/2018)

layersizes = model.layersizes;
numlayers = model.numlayers;
noProps = 1;
maxNoProps = Inf;
n = size(X,2);
sz = floor(0.05*n);
psize = layersizes(1,2:(numlayers+1))*layersizes(1,1:numlayers)' + sum(layersizes(2:(numlayers+1)));

alpha = 0.005;
beta = 0.9;

max_iters = 100;
cur = 0;
cur_time = 0;
name = 'sgd';

if isfield(options, 'alpha')
  alpha = options.alpha;
end
if isfield(options, 'beta')
  beta = options.beta;
end

if isfield(options,'hs')
  sz = options.hs;
end

if isfield(options, 'maxNoProps')
    maxNoProps  = options.maxNoProps;
end 


if isfield(options,'max_iters')
  max_iters = options.max_iters;
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
    %maxNoProps = maxNoProps + noProps;
    
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
fprintf(' batch size: %d\n step size: %f\n momentum: %f\n max props: %d\n\n',...
    sz, alpha, beta, maxNoProps);

tic;
% training
fprintf('\n start training...\n');
momentum_params = 0;
for iter = cur+1: cur + max_iters
   if noProps > maxNoProps
        iter = iter - 1;
        break;
   end
   idx = randsample(n, sz);
   x_sample = X(:,idx);
   y_sample = y(:,idx);
   [~, grad] = compute_model(model, params, x_sample, y_sample);
   grad = grad + lambda*params;
   noProps = noProps + 2*size(x_sample,2);
   ll_err = compute_model(model, params, X,y);
   ll = ll_err(1); tr_err = ll_err(2);
   tr_loss = ll + 0.5 * lambda * (params'*params);
   
   te_loss_err = compute_model(model, params, X_test, y_test);
   te_loss = te_loss_err(1); te_err = te_loss_err(2);
   
   options.tr_losses(iter) = tr_loss;
   options.tr_errs(iter) = tr_err;
   options.te_losses(iter) = te_loss;
   options.te_errs(iter) = te_err;
   options.tr_grad(iter) = norm(grad,Inf);
   options.tr_times(iter) = toc + cur_time;
   options.tr_noProps(iter) = noProps;
   fprintf('\nIter: %d, time = %f s\n', iter, options.tr_times(iter));
   fprintf('training loss + reg: %f, grad: %f(max), %f(norm)\n', tr_loss, norm(grad,Inf), norm(grad,2));
   fprintf('training err: %f\n', tr_err);
   fprintf('test loss: %f, test err: %f\n', te_loss, te_err);
   fprintf('total MVPs: %g, total Props: %g\n', 0, noProps);
   % 
   momentum_params = beta * momentum_params - alpha *grad;
   params = params + momentum_params;
   
    
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
