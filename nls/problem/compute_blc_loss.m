function l = compute_blc_loss(X,y, w)
% X: n x d matrix
% y: n x 1 array
% w: d x 1 array
% a: n x 1 array
l = mean((y - sigmoid(X*w)).^2);
end
