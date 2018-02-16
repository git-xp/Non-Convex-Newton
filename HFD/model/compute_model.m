function [loss,grad,hess,perr] = compute_model(model,params,X,y)
%% Evaluting neural network models: loss, grad, Hess(), error
% Input:
%       model       ---- specification neural network model
%         model.numlayers: number of layers
%         model.layersizes: number of neurons of each layer
%         model.layertypes: type of each layer (currently only support
%                           logisitic(sigmoid),tanh,linear,softmax)
%       params      ---- weights
%       X,y         ---- input(d x n),output(c x n), d,c should be
%                        consistent with model.layersizes.
%
% Output:
%       loss        ---- loss 
%       grad        ---- gradient, same size as params
%       hess        ---- a function compute Hessian-vector product.
%       perr        ---- learning error,e.g. classification error or mse.
%    if only returning 1 output, it returns [loss, learning error]
%     i.e. res = compute_model(model,params,X,y) --> res = [loss, err] 
%    if returning 2 outputs,  it returns [[loss,err], gradient]
%     i.e. [res,grad] = compute_model(model,params,X,y) -> res = [loss, err]
%    if returning 3 outputs, it returns [loss, grad, hess]
%    
% written by Peng Xu, Fred Roosta, 6/8/2017, updated(2/8/2018)

n = size(X,2);
numlayers = model.numlayers;
layertypes = model.layertypes;
layersizes = model.layersizes;
psize = length(params);
dW = cell(numlayers, 1);
db = cell(numlayers, 1);



[W,b] = unpack(params, numlayers, layersizes);

xi = X;
z = cell(numlayers+1,1);
dx = cell(numlayers,1);
z{1} = X;

ll = 0;

%% loss
for i = 1:numlayers
    xi = bsxfun(@plus, W{i}*z{i}, b{i});
    if strcmp(layertypes{i}, 'logistic');
        zi = 1./(1 + exp(-xi));
    elseif strcmp(layertypes{i}, 'tanh');
        zi = tanh(xi);
    elseif strcmp(layertypes{i}, 'linear');
        zi = xi;
    elseif strcmp(layertypes{i}, 'softmax');
        zi = softmax(xi);
    else
        error('Unknow layer type');
    end
    z{i+1} = zi;
end

if strcmp(layertypes{numlayers}, 'linear')
    ll = ll - sum(sum((y - zi).^2));
    err = 2*(y-zi);
elseif strcmp( layertypes{numlayers}, 'logistic' )
%     ll = ll + double( sum(sum(y.*log(xi + (y==0)) + (1-y).*log(1-xi + (y==1)))) );
    % more stable:
    ll = ll + sum(sum(xi.*(y - (xi >= 0)) - log(1+exp(xi - 2*xi.*(xi>=0)))));                
    err = (y - zi);
elseif strcmp( layertypes{numlayers}, 'softmax' )
    ll = ll + double(sum(sum(y.*log(zi))));
    err = y - zi;
end

loss = -ll/n;

if nargout <=2 
    % compute error
    if strcmp(model.type, 'mse')
        perr = sum(sum((y - zi).^2))/n;
    elseif strcmp(model.type, 'classification')
        [~,labels] = max(zi);
        [~,truelabels] = max(y);
        perr = mean(truelabels ~= labels);
    end
    loss = [loss; perr];
    if nargout == 1
        return;
    end
end

%% gradient
db{numlayers} = sum(err,2);
dW{numlayers} = err * z{numlayers}';
dx{numlayers} = err;
err = W{numlayers}' * err;
for i = numlayers-1:-1:1
    xi = z{i+1};
    if strcmp(layertypes{i}, 'logistic');
        err = err.* (1-xi).*xi;
    elseif strcmp(layertypes{i}, 'tanh');
        err = err .* (1 - xi) .* (1 + xi);
    elseif strcmp(layertypes{i}, 'linear');
        % err = err;
%     elseif strcmp(layertypes{i}, 'softmax');
%         xi = exp(xi);
%         xi = bsxfun(@rdivide, xi, sum(xi,2));
    else
        error('Unknow layer type');
    end
%     z{i+1} = [];
    db{i} = sum(err,2);
    dW{i} = err * z{i}';
    dx{i} = err;
    err = W{i}'*err;
end
grad = pack(dW, db, psize, numlayers, layersizes);
grad = -grad/n;

if nargout < 3
    return;
end


%% compute Hessian 

    function HV = compute_HV(V)
        RdW = cell(numlayers, 1);
        Rdb = cell(numlayers, 1);
        Rdz = cell(numlayers, 1);
        Rdx = cell(numlayers, 1);
        Rz = cell(numlayers+1,1);
        Rz{1} = zeros(size(X));
        Rx = cell(numlayers+1,1);
        [VW,Vb] = unpack(V, numlayers, layersizes);
        for i = 1:numlayers
            xi = bsxfun(@plus, W{i}*z{i}, b{i});
            rxi = W{i}*Rz{i} + bsxfun(@plus, VW{i}*z{i}, Vb{i});
            if strcmp(layertypes{i}, 'logistic');
                rzi = rxi.*(1-z{i+1}).*z{i+1};
            elseif strcmp(layertypes{i}, 'tanh');
                rzi = rxi.*(1-z{i+1}).*(1 + z{i+1});
            elseif strcmp(layertypes{i}, 'linear');
                rzi = rxi;
            elseif strcmp(layertypes{i}, 'softmax');
                rzi = z{i+1}.*rxi;
                rzi = rzi - bsxfun(@times, z{i+1},sum(rzi,1));
            else
                error('Unknow layer type');
            end
            Rz{i+1} = rzi;
            Rx{i+1} = rxi;
        end
        % err = dE/dx

        if strcmp(layertypes{numlayers}, 'linear')
            error('Unknow layer type');
        elseif strcmp( layertypes{numlayers}, 'logistic' )
            Rdx{numlayers} = - Rz{numlayers+1};
%             Rdx{numlayers} = -(1 - z{numlayers+1}).*z{numlayers+1} .*Rx{numlayers+1};
        elseif strcmp( layertypes{numlayers}, 'softmax' )
            Rdx{numlayers} = -Rz{numlayers+1};
        else
            error('Unknow layer type');
        end
        RdW{numlayers} = Rdx{numlayers} * z{numlayers}' + dx{numlayers} * Rz{numlayers}';
        Rdb{numlayers} = sum(Rdx{numlayers},2);
        Rdz{numlayers} = VW{numlayers}'*dx{numlayers} + W{numlayers}'*Rdx{numlayers};
        for i = numlayers-1:-1:1
            if strcmp(layertypes{i}, 'logistic');
                Rdx{i} = (1 - z{i+1}).*z{i+1}.*Rdz{i+1} + Rx{i+1}.*(1 - 2*z{i+1}).*dx{i};
            elseif strcmp(layertypes{i}, 'tanh');

            elseif strcmp(layertypes{i}, 'linear');
                Rdx{i} = Rdz{i+1};
            else
                error('Unknow layer type');
            end
            RdW{i} = Rdx{i} * z{i}' + dx{i} * Rz{i}';
            Rdb{i} = sum(Rdx{i},2);
            Rdz{i} = VW{i}'*dx{i} + W{i}'*Rdx{i};
        end
        HV = pack(RdW, Rdb, psize, numlayers, layersizes);
        HV = HV/n;
    end


hess = @(V)-compute_HV(V);

if nargout < 4
    return
end
% compute error
if strcmp(model.type, 'mse')
    perr = sum(sum((y - zi).^2))/n;
elseif strcmp(model.type, 'classification')
    [~,labels] = max(zi);
    [~,truelabels] = max(y);
    perr = mean(truelabels ~= labels);
end
end


function M = pack(W,b, psize, numlayers, layersizes)
    
    M = zeros( psize, 1 );
    
    cur = 0;
    for i = 1:numlayers
        M((cur+1):(cur + layersizes(i)*layersizes(i+1)), 1) = vec( W{i} );
        cur = cur + layersizes(i)*layersizes(i+1);
        
        M((cur+1):(cur + layersizes(i+1)), 1) = vec( b{i} );
        cur = cur + layersizes(i+1);
    end
    
end

function [W,b] = unpack(M, numlayers, layersizes)

    W = cell( numlayers, 1 );
    b = cell( numlayers, 1 );
    
    cur = 0;
    for i = 1:numlayers
        W{i} = reshape( M((cur+1):(cur + layersizes(i)*layersizes(i+1)), 1), [layersizes(i+1) layersizes(i)] );

        cur = cur + layersizes(i)*layersizes(i+1);
        
        b{i} = reshape( M((cur+1):(cur + layersizes(i+1)), 1), [layersizes(i+1) 1] );

        cur = cur + layersizes(i+1);
    end
    
end

function v = vec(A)

v = A(:);
end