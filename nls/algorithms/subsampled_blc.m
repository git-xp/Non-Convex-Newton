function [sol, results] = subsampled_blc(X,Y,problem,params)
% subsample newton solver with full gradient for the following problem:
%       min_w sum_i l(w'*x_i,y_i) + lambda*norm(w)^2
% In this function, we have TR, ARC, GN and their subsampled variants. 
% LBFGS, GD and AGD are also included. 
%
% input:
%       X,Y                     ---- data matrices
%       problem:
%           problem.loss        ---- loss, get_grad, get_hessian
%           problem.grad        ---- function to compute gradient
%           problem.hessian     ---- function to compute the diagonal D^2
%           problem.lambda      ---- ridge parameters
%           problem.w0          ---- initial start point (optional)
%           problem.w_opt       ---- optimal solution (optinal)
%           problem.l_opt       ---- minimum loss (optional)
%           problem.condition   ---- condition number (optional)
%       params:
%           params.method       ---- method names
%           params.hessian_size ---- sketching size for hessian
%           params.step_size    ---- step size
%           params.niter        ---- number of outer iterations
%           params.sigma        ---- cubic parameter of ARC
%           params.delta        ---- trust-region radius of TR
%           params.inner_iters  ---- maximum inner iterations
%           params.solver       ---- algorithms for subproblem
%           params.line_search  ---- whether using line search or not
%           params.eta1,eta2,gamma1,gamma2
%                               ---- parameters for adaptive schemes
%           params.max_props    ---- maximum numbero of propagations
%
% output:
%       sol ---- final solution
%       results:
%           results.t       ---- running time at every iteration
%           results.noProps ---- number of propagations at every iteration
%           results.grads   ---- gradient norm at every iteration
%           results.l       ---- objective value
%           results.err     ---- solution error (if problem.w_opt given)
%           results.sol     ---- solution at every iteration
%
%
%
% written by Peng Xu, Fred Roosta,  6/4/2017, updated(2/8/2018)



if nargin == 3
    params = 0;
end

loss = problem.loss;
get_grad = problem.grad;
get_hessian = problem.hessian;

[n,d] = size(X);

% default setting
lambda = 0;
method= 'Uniform';
niters = 1e4;            % total number of iterations
inner_iters = 250;      % number of inner iteration for subproblems.
linesearch = false;
eta = 1;                % step size
w0 = zeros(d,1);        % initial point
delta = 1;
max_delta = Inf;
sigma = 1/delta;
min_sigma = 0;
eta1 = 0.8;
eta2 = 1e-4;
gamma1 = 2;
gamma2 = 1.2;
solver = 'Steighaug';
max_props = 1e9;

% check params
if isfield(problem, 'lambda')
    lambda = problem.lambda;
end

if isfield(problem, 'w0')
    w0 = problem.w0;
end

if isfield(params, 'method')
    method = params.method;
end

if isfield(params, 'hessian_size')
    s = params.hessian_size;
end

if isfield(params, 'step_size')
    eta = params.step_size;
end

if isfield(params,'niters')
    niters = params.niters;
end

if isfield(params,'inner_iters')
    inner_iters = params.inner_iters;
end

if isfield(params,'max_props')
    max_props = params.max_props;
end

if isfield(params, 'beta0')
    beta0 = params.beta0;
end

if isfield(params, 'delta')
    delta = params.delta;
end
if isfield(params, 'max_delta')
    max_delta = params.max_delta;
end

if isfield(params, 'sigma')
    sigma = params.sigma;
end
if isfield(params, 'min_sigma')
    min_sigma = params.min_sigma;
end


if isfield(params, 'eta1')
    eta1 = params.eta1;
    gamma1 = params.gamma1;
end

if isfield(params,'eta2')
    eta2 = params.eta2;
    gamma2 = params.gamma2;
end

if isfield(params,'solver')
    solver = params.solver;
end

if isfield(params,'linesearch')
    linesearch = params.linesearch;
    eta0 = eta;
end

w = w0;
t = zeros(niters,1);
sol = zeros(d, niters);
noProps = zeros(1,niters);
nopProps_sofar = 1;

% algorithm start
fprintf('algorithm start ......\n');
tic;

if strcmp(method, 'RNS')
    rnorms = sum(X.^2,2);
end
if strcmp(method, 'SGD')
    schedulers = randsample(n,niters,'true');
end

rnorms = sum(X.^2,2);

for i = 1:niters
    
    % compute hessian
    switch method
        case {'Newton-TR', 'Newton-ARC'}
            D2 = get_hessian(X,Y,w);
            H = X'*bsxfun(@times, D2, X) + lambda*eye(d);H = (H + H')/2;
            hessianSamplesSize = size(X,1);
        case 'GN'
            D2 = problem.get_gn(X,Y,w);
            H = X'*bsxfun(@times, D2, X) + lambda*eye(d);H = (H + H')/2;
            hessianSamplesSize = size(X,1);
        case 'RNS-GN'
            D2 = problem.get_gn(X,Y,w);
            p = abs(D2).*rnorms;
            p = full(p/sum(p));

            q = min(1,p*s);idx = rand(n,1)<q;p_sub = q(idx);
            X_sub = X(idx,:); D2_sub = D2(idx);
            H = X_sub'*bsxfun(@times,D2_sub./p_sub,X_sub)+lambda*eye(d);H = (H + H')/2;
            hessianSamplesSize = size(X_sub,1);
        case 'Uniform-GN'
            XY = datasample([X,Y], s, 1, 'Replace', false);
            X_sub = XY(:,1:end-1);
            D2_sub = problem.get_gn(X_sub,XY(:,end),w);
            H = X_sub'*bsxfun(@times,D2_sub,X_sub)+lambda*eye(d);H = (H + H')/2;
            hessianSamplesSize = size(X_sub,1);
        case {'RNS-TR','RNS-ARC'}
            D2 = get_hessian(X,Y,w);
            p = abs(D2).*rnorms;
            p = full(p/sum(p));
            %idx = randsample(n,s,true,p); p_sub = p(idx);
            %q = min(1,p*s);idx = rand(n,1)<q;p_sub = q(idx);
            idx = datasample(1:n,s,'Weight',p);p_sub = p(idx)*s;
            X_sub = X(idx,:); D2_sub = D2(idx);
            H = X_sub'*bsxfun(@times,D2_sub./p_sub,X_sub)+lambda*eye(d);H = (H + H')/2;
            hessianSamplesSize = size(X_sub,1);
            
        case {'Uniform-TR', 'Uniform-ARC'}
            % idx = randsample(n,s); % default is sampling without replacement
            % X_sub = X(idx,:);
            % D2_sub = get_hessian(X_sub,Y(idx),w);
            XY = datasample([X,Y], s, 1, 'Replace', false);
            X_sub = XY(:,1:end-1);
            D2_sub = get_hessian(X_sub,XY(:,end),w);
            H = X_sub'*bsxfun(@times,D2_sub,X_sub)+lambda*eye(d);H = (H + H')/2;
            hessianSamplesSize = size(X_sub,1);
    end
    

    
    if strcmp(method, 'SGD')
        datax = X(schedulers(i),:);
        z = -(datax'*(get_grad(datax,Y(schedulers(i)),w)) + lambda*w/n);
        eta = params.step_size/(1+i/5000);
        w = w + eta*z;
        nopProps_sofar = nopProps_sofar + 2*size(datax,1);
    elseif strcmp(method,'AGD')
        c = get_grad(X,Y,w);
        nopProps_sofar = nopProps_sofar + 2*size(X,1);
        grad = X'*c + lambda*w;
        if i == 1
            ys = w;
        end
        ys1 = w - eta*grad;
        w = (1+beta0)*ys1 - beta0*ys;
        ys = ys1;
        
    else
        c = get_grad(X,Y,w);
        grad = X'*c + lambda*w;
        nopProps_sofar = nopProps_sofar + 2*size(X,1);
        switch method
            case 'LBFGS'
                if i == 1
                    H = eye(d);
                    z = -grad;
                    y = zeros(d,0);
                    S = zeros(d,0);
                else
                    s_prev = alpha_prev*z_prev;
                    y_prev = grad - v_prev;
                    
                    if size(S,2) >=  params.L
                        S = [S(:,2:end),s_prev];
                        y = [y(:,2:end),y_prev];
                    else
                        S = [S, s_prev];
                        y = [y, y_prev];
                    end
                    z = -lbfgs(grad,S,y,speye(d));
                    
                end
                z_prev = z;
                v_prev = grad;
                
            case {'GN', 'RNS-GN', 'Uniform-GN'}
                [z,~,~,iter] = pcg(H, -grad);
                nopProps_sofar = nopProps_sofar + iter*2*hessianSamplesSize;
            case 'GD'
                z = -grad;
                
            case {'RNS-TR','Uniform-TR', 'Newton-TR'}
                assert(eta == 1);
                fail_count = 0;
                tr_loss = loss(X,Y,w) + lambda * norm(w)^2;
                % nopProps_sofar = nopProps_sofar + size(X,1); % Already
                % counted in gradient
                % fprintf('\nEpoch: %d, time = %g s\n', i, toc);
                % fprintf('training loss + reg: %g, grad: %g(max), %g(norm)\n', tr_loss, norm(grad,Inf), norm(grad,2));
                % fprintf('training err: %g\n', tr_err);
                
                while true
                    switch solver 
                        case 'Steihaug'
                            steihaugParams = [1e-9, inner_iters, 0]; % parameters for Steighaug-CG
                            if fail_count == 0
                                z0 = randn(d,1);
                                z0 = 0.99*delta*z0/norm(z0);
                            else
                                z0 = [];
                            end
                            [z,m, num, iflag] = cg_steihaug (@(x)H*x, grad, delta, steihaugParams, [] );
                            nopProps_sofar = nopProps_sofar + num*2*hessianSamplesSize;
                        case 'Others'
                            fprint('Not Implemented!\n Please use Steihaug!\n');
                            return;
                    end

                    assert(m <= 0);
                    newll = loss(X,Y, w + z);
                    newll = newll + 0.5 * lambda * norm(w + z)^2;
                    nopProps_sofar = nopProps_sofar + size(X,1);
                    rho = (tr_loss - newll)/-m;
                    if rho < eta2
                        fail_count = fail_count + 1;
                        % fprintf('FALIURE No. %d: delta = %g, rho = %g, iters: %g\n', fail_count, delta, rho,num);
                        delta = delta/gamma1;
                        z = 0;
                    elseif rho < eta1
                        % fprintf('SUCCESS: delta = %g, rho = %g, s = %g\niters: %g\n', delta, rho, norm(z), num );
                        %                             w = w + z;
                        delta = min(max_delta, gamma2*delta);
                        break;
                    else
                        % fprintf('SUPER SUCCESS: delta = %g, rho = %g, s = %g\niters: %g\n', delta, rho, norm(z),num );
                        %                             w = w + z;
                        delta = min(max_delta, gamma1*delta);
                        break;
                    end
                end
                
            case {'Uniform-ARC','RNS-ARC','Newton-ARC'}
                assert(eta == 1);
                fail_count = 0;
                tr_loss = loss(X,Y,w) + lambda * norm(w)^2;
                % nopProps_sofar = nopProps_sofar + size(X,1);% Already
                % counted in gradient
                % fprintf('\nEpoch: %d, time = %g s\n', i, toc);
                % fprintf('training loss + reg: %g, grad: %g(max), %g(norm)\n', tr_loss, norm(grad,Inf), norm(grad,2));
                % fprintf('training err: %g\n', tr_err);

                while true
                    if norm(grad) == 0
                        grad = eps*randn(d,1);
                    end
                    [z,m, num] = cubic_solver(H,grad,sigma,inner_iters,1e-12);
                    nopProps_sofar = nopProps_sofar + num*2*hessianSamplesSize;
                    if m >= 0 % ask for next Hessian.
                        break;
                    end
                    newll = loss(X,Y, w + z);
                    newll = newll + 0.5 * lambda * norm(w + z)^2;
                    nopProps_sofar = nopProps_sofar + size(X,1);
                    rho = (tr_loss - newll)/-m;
                    if rho == 0 && m <= eps*norm(grad) % deal with numerical issuses
                        sigma = max(min_sigma, sigma/gamma1);
                        break;
                    end
                    if rho < eta2 || isnan(rho)
                        sigma = sigma*gamma1;
                        % fprintf('Fail: sigma is now %g\n',sigma);
                        fail_count = fail_count + 1;
                        z = 0;
                    elseif rho < eta1
                        sigma = max(min_sigma, sigma/gamma2);
                        break;
                    else
                        sigma = max(min_sigma, sigma/gamma1);
                        break;
                    end
                    
                    
                end
                
        end
        if linesearch
            eta = eta0;
            l = loss(X,Y,w);
            l = l + 0.5 * lambda * norm(w)^2;
            % nopProps_sofar = nopProps_sofar + size(X,1);% Already
            % counted in gradient
            deltaf = params.alpha * z' * grad;
            max_count = 0;
            while ( ( loss(X,Y,w + eta*z) + 0.5 * lambda * norm(w + eta*z)^2 ) >= ( l + deltaf * eta) ) && (max_count < 20)
                eta = eta * params.beta;
                max_count = max_count + 1;
                nopProps_sofar = nopProps_sofar + size(X,1);
            end

        end
        alpha_prev = eta;
        w = w + eta*z;
    end
    sol(:,i) = w;
    t(i) = toc;
    noProps(i) = nopProps_sofar;
    if nopProps_sofar >= max_props
        break;
    end
end
fprintf('main algorithm end\n');
iters = i;
% better improve this using vector operations
fprintf('Further postprocessing......\n')
t = [0;t(1:iters)];
sol = [w0,sol(:,1:iters)];
results.t = t;
results.sol = sol;
results.noProps = [1,noProps(1:iters)];
grads = zeros(iters+1,1);
l = zeros(iters+1,1);
for i = 1:iters+1
    w = sol(:,i);
    l(i) = (loss(X,Y,w) + lambda*(w'*w)/2);
    c = get_grad(X,Y,w);
    grad = X'*c + lambda*w;
    grads(i) = norm(grad);
end
results.l = l;
results.grads = grads;
if isfield(problem, 'w_opt')
    w_opt = problem.w_opt;
    err = bsxfun(@minus, sol, w_opt);
    err = sqrt(sum(err.*err))';
    results.err = err;%/norm(w_opt,2);
end

if isfield(params,'name')
    results.name = params.name;
end
fprintf('DONE! :) \n');
end

