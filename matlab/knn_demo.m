% Simple KNN demo in MATLAB/Octave
% Loads the same CSV used by Python and does a train/test split.
% No toolbox is required; this is a tiny from-scratch KNN (k=5).

clear; clc;

data = readtable('../data/iris.csv');
features = data{:, {'sepal_length','sepal_width','petal_length','petal_width'}};
labels = data.target;

% encode class labels to ints 1..C and back
[ulabs, ~, yint] = unique(labels);
X = features;
y = yint;

% train/test split
rng(42);
n = size(X,1);
idx = randperm(n);
cut = floor(0.8*n);
tr = idx(1:cut);
te = idx(cut+1:end);

Xtr = X(tr,:); ytr = y(tr);
Xte = X(te,:); yte = y(te);

% z-score normalize using train stats
mu = mean(Xtr,1); sig = std(Xtr,[],1); sig(sig==0)=1;
Xtr = (Xtr - mu)./sig;
Xte = (Xte - mu)./sig;

k = 5;
pred = zeros(length(yte),1);

for i = 1:length(yte)
    xi = Xte(i,:);
    dists = sqrt(sum((Xtr - xi).^2, 2)); % Euclidean
    [~, order] = sort(dists, 'ascend');
    kneigh = ytr(order(1:k));
    % majority vote
    pred(i) = mode(kneigh);
end

acc = mean(pred == yte);
fprintf('KNN (k=%d) accuracy: %.3f\n', k, acc);

% confusion matrix
C = confusionmat(yte, pred);
disp('Confusion matrix (rows=true, cols=pred):');
disp(C);

% show mapping back to class names
disp('Label mapping:');
for i=1:numel(ulabs)
    fprintf('%d -> %s\n', i, string(ulabs(i)));
end
