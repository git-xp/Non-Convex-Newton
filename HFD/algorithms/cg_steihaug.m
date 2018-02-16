%--------------------------------------------------------------------------
%
function [ p,m, num_cg, iflag ] = cg_steihaug (  H, g, delta, params, x0 )
%
%--------------------------------------------------------------------------
%
% ... This procedure approximately solves the following trust region
%     problem
%
%         minimize    m(p) = 1/2 p'Hp + g'p
%         subject to  ||p|| <= Delta
%
%
%     by means of the CG-Steihaug method.
%
%--------------------------------------------------------------------------
% INPUT
%
%             H:  Hessian matrix
%             g:  gradient vector
%         delta:  radius of the TR
%     params(1):  relative residual reduction factor
%     params(2):  max number of iterations
%     params(3):  level of output
%
% OUTPUT
%             p:  an aproximate solution of (1)-(2)
%        num_cg:  number of CG iterations to achieve convergence
%         iflag:  termination condition
%
%--------------------------------------------------------------------------
%

tr_model = @(x) 0.5*x'* H(x) + x'*g;
n      = length(g);
errtol = params(1); maxit  = params(2); iprnt  = params(3); iflag  = ' ';
%
x  = zeros(n,1); 
if ~isempty(x0)
    x = x0;
end
r = -g - H(x);
z   = r;
rho = z'*r;
tst = norm(r);
flag  = '';
terminate = errtol*norm(r);   it = 1;    hatdel = delta*1;%(1-1.d-6);
rhoold = 1.0d0;
if iprnt > 0
    fprintf(1,'\n\tThis is an output of the CG-Steihaug method. \n\tDelta = %7.1e \n', delta);
    fprintf(1,'   ---------------------------------------------\n');
end
flag = 'We do not know ';
if tst <= terminate; flag  = 'Small ||g||    '; end

while((tst > terminate) && (it <=  maxit) && norm(x) <=  hatdel)
    %
    if(it == 1)
        p = z;
    else
        beta = rho/rhoold;
        p = z + beta*p;
    end
    w  = H(p);  alpha = w'*p;
    ineg = 0;
    if(alpha <=  0)
        ac = p'*p; bc = 2*(x'*p); cc = x'*x - delta*delta;
        alpha = (-bc + sqrt(bc*bc - 4*ac*cc))/(2*ac);
        flag  = 'negative curvature';
        iflag = 'NC';
        x = x + alpha * p;
        break;
    else
        alpha = rho/alpha;
        if norm(x+alpha*p) > delta
            ac = p'*p; bc = 2*(x'*p); cc = x'*x - delta*delta;
            alpha = (-bc + sqrt(bc*bc - 4*ac*cc))/(2*ac);
            flag  = 'boundary was hit';
            iflag = 'TR';
            x = x + alpha * p;
            break;
        end
    end
    x   =  x + alpha*p;
    r   =  r - alpha*w;
    tst = norm(r);
    if tst <= terminate; flag = '||r|| < test   '; iflag = 'RS'; break; end;
    if norm(x) >=  hatdel; flag = 'close to the boundary'; iflag = 'TR'; break;end
    
    if iprnt > 0
        fprintf(1,' %3i    %14.8e   %s  \n', it, tst, flag);
    end
    rhoold = rho;
    z   = r;
    rho = z'*r;
    it  = it + 1;
end %
if it > maxit; iflag = 'MX'; end;

num_cg = it;
p = x;
m = tr_model(p);
%
