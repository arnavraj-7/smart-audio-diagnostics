function [maxErr, medErr] = sa_verify(wavFile, domain, refCsv, doDenoise)
% Cross-toolchain check: computes the 53-feature vector in MATLAB and compares
% it with the Python vector for the same file, written by
%   python scripts/export_for_matlab.py --verify <wavFile> --domain <domain>
% Confirms that the MATLAB and Python DSP front ends agree.
if nargin < 3
    refCsv = fullfile(fileparts(mfilename('fullpath')), '..', ...
                      'results', 'tables', 'verify_python.csv');
end
if nargin < 4, doDenoise = false; end
cfg = sa_config(domain);
[x, srIn] = audioread(wavFile);
[y, sr]   = sa_preprocess(x, srIn, cfg, doDenoise);
[v, names] = sa_features(y, sr, cfg);

ref = readmatrix(refCsv);
n = min(numel(v), numel(ref));
relErr = abs(v(1:n) - ref(1:n).') ./ (abs(ref(1:n).') + 1e-6);

fprintf('%-28s %14s %14s %10s\n', 'Feature', 'MATLAB', 'Python', 'rel.err');
for i = 1:n
    fprintf('%-28s %14.6f %14.6f %10.2e\n', names{i}, v(i), ref(i), relErr(i));
end
maxErr = max(relErr); medErr = median(relErr);
fprintf('\nmax relative error = %.3e   (median %.3e)\n', maxErr, medErr);
end
