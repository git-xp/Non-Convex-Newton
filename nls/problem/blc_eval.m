function acc = blc_eval(X,Y, w)
% This is correct but requires large memory 
% Yhat = sigmoid(X*w) > 0.5;
% corrects = bsxfun(@(x,y)x==y,Yhat, Y);
% acc = mean(corrects);
% acc = acc';

K = size(w,2);
acc = zeros(K,1);
for i = 1:K
    Yhat = sigmoid(X*(w(:,i))) > 0.5;
    corrects = bsxfun(@(x,y)x==y,Yhat, Y);
    acc(i) = mean(corrects);
end
end