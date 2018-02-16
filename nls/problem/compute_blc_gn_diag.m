function d = compute_blc_gn_diag(X,y,w)
% X: n x d matrix
% y: n x 1 array
% w: d x 1 array
% a: n x 1 array
% d: n x 1 array
n = length(y);
a = sigmoid(X*w);
d = 2/n * (a.^2 .*(1 - a).^2);
end