function g = compute_blc_gradient(X,y,w)
% X: n x d matrix
% y: n x 1 array
% w: d x 1 array
% a: n x 1 array
% g: n x 1 array
n = length(y);
a = sigmoid(X*w);
g = 2/n *((a - y).*a.*(1-a));
end