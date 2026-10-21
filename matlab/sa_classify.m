function [label, confidence, scores] = sa_classify(v, domain, modelDir)
% Applies the model exported from Python (scripts/export_for_matlab.py).
% k-NN is used on the MATLAB side because it needs no toolbox and no
% re-implementation of the trained SVM/RF - the training data and the
% standardisation come straight from the Python run.
if nargin < 3, modelDir = fullfile(fileparts(mfilename('fullpath')), '..', 'model_export'); end

persistent CACHE
if isempty(CACHE), CACHE = struct(); end
key = matlab.lang.makeValidName(domain);

if ~isfield(CACHE, key)
    m.mu    = readmatrix(fullfile(modelDir, [domain '_scaler_mean.csv']));
    m.sigma = readmatrix(fullfile(modelDir, [domain '_scaler_scale.csv']));
    m.X     = readmatrix(fullfile(modelDir, [domain '_train_X.csv']));
    T       = readtable(fullfile(modelDir, [domain '_train_y.csv']), ...
                        'TextType','string','ReadVariableNames',true);
    m.y     = string(T{:,1});
    m.k     = 5;
    CACHE.(key) = m;
end
m = CACHE.(key);

z = (v(:).' - m.mu(:).') ./ max(m.sigma(:).', 1e-12);
d = sqrt(sum((m.X - z).^2, 2));
[~, ord] = sort(d);
nn = m.y(ord(1:m.k));

classes = unique(m.y);
scores = zeros(numel(classes),1);
for i = 1:numel(classes), scores(i) = sum(nn == classes(i)) / m.k; end
[confidence, bi] = max(scores);
label = char(classes(bi));
scores = table(classes, scores, 'VariableNames', {'Class','Score'});
end
