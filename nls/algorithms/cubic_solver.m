function [z,m,num] = cubic_solver(H,g,sigma,max_iters,tol)
% Generialzied Lanczos method for solving the cubic regularization
% subproblem:
%           minimize g'x + 1/2 x'Hx + sigma/3 x'Hx
% This is different implementation from the original algorithm. Here we
% first run all the lanczos iteration then solve the low-dimensional cubic
% problem with a tridiagonal matrix.
%
% input:
%       H,g,sigma           ---- cubic problem
%       max_iters           ---- number of Lanczos iterations
%       tol                 ---- error tolarance
% output:
%       z                   ---- solution
%       m                   ---- objective value
%       num                 ---- number of Hessian-vector product
%
% written by Peng Xu, Fred Roosta, 6/4/2017, updated(2/8/2018)

d = size(g,1);
K = min(d, max_iters);
Q = zeros(d,K);
% q = randn(d,1); q = q/norm(q);
q = g+randn(d,1)/sqrt(d); q = q/norm(q);
% q = g + randn(d,1); q = q/norm(q);
T = zeros(K+1,K+1);
beta = 0;
q0 = 0;
tol = min(tol, tol*norm(g));

for i = 1:K
    Q(:,i) = q;
    v =H*q;% H(q);
    alpha = q' * v;
    T(i,i) = alpha;
    % r = v - beta * q0 - T(i,i)*q;
    % r = get_residue(v, Q(:,1:i));
    r = v - Q(:,1:i) * (Q(:,1:i)' * v); % reothorgonalization
    beta = norm(r);
    T(i,i+1) = beta;
    T(i+1,i) = beta;

    if beta < tol
        % fprintf('low rank!\n')
        break;
    end
    q0 = q;
    q = r/beta;
end
% fprintf('lanczos stops at %d iteration\n', i);
T = T(1:i,1:i);
Q = Q(:,1:i);
num = i;

if norm(T) < tol && norm(g) < eps
    z = zeros(d,1);
    m = 0;
    return;
end

gt = Q'*g;



options = optimoptions(@fminunc,'SpecifyObjectiveGradient',true,...
   'Algorithm','quasi-newton','Display','off','OptimalityTolerance',tol);
z0 = zeros(i,1);
%% Here is to deal with numerical issues
try
    [z, m, flag, output] = fminunc(@(z)cubic_prob(T,gt,sigma, z), -gt ,options);
catch
    try
        [z, m, flag, output] = fminunc(@(z)cubic_prob(T,gt,sigma, z), z0 ,options);
    catch
        [z, m, flag, output] = fminunc(@(z)cubic_prob(T,gt,sigma, z), randn(i,1),options);
    end
end
z = Q*z;

end

function [f,g,H] = cubic_prob(Hess,grad,sigma,z)
znorm = norm(z);
f = grad'*z + 1/2 * z'*Hess*z + 1/3*sigma*znorm^3;
% f = grad'*z + 1/2 * z'*Hess(z) + 1/3*sigma*norm(z)^3;
g = grad + Hess*z + sigma * znorm *z;
H = Hess + sigma * (z*z'/znorm + eye(size(z,1))*znorm);
end

function hv = HvFunc(v,x, Hess,sigma)
znorm = norm(x);
hv = Hess(v) + sigma*(x'*v/znorm*x + znorm*v);
end


