function labels_01 = getClassLabels(labels_raw, lastClass)
%%
% Gets in arbitrary labels and returns vectors with 0's and 1's where
% for each data point, we have labels_01(i,j) = 1 if data point i belongs
% to class j.
% @lastClass: a true/flase variable dtermining whether the label of the
% last class is omitted.
%       - For training, set lastClass = true
%         the label for the last class is recoverable from the % other classes,
%         i.e., probability of a data point belonging to a class must be 1.
%       - For training set lastClass = false (default)

if nargin == 1
    lastClass = false;
end
fprintf('Generating 0-1 labels....labels for last class omitted: %g...',lastClass);
class_labels = sort(unique(labels_raw));
nData_train = size(labels_raw,1);
nClasses = length(class_labels);
labels_train = zeros(nData_train,1);
for i=1:nClasses
    ind = labels_raw == class_labels(i);
    labels_train(ind,:) = i;
end
labels_01 = sparse( 1:nData_train, labels_train, ones( nData_train, 1 ));
if lastClass
    labels_01 = labels_01(:,1:end-1);
else
    labels_01 = full(labels_01);
end
%% Sanity check
ubbb = sort(unique(labels_01));
if nClasses > 1
    assert(length(ubbb) == 2);
    assert(ubbb(1) == 0 && ubbb(2) == 1);
else
    assert(length(ubbb) == 1);
    assert(ubbb(1) == 1);
end
fprintf('Done\n');
end