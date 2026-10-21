function [y, sr] = sa_preprocess(x, srIn, cfg, doDenoise)
% M2 pre-processing chain, matching src/preprocess.py step for step.
if nargin < 4, doDenoise = true; end

x = double(x(:));
if size(x,2) > 1, x = mean(x,2); end

sr = cfg.sr;
if srIn ~= sr
    g = gcd(round(srIn), round(sr));
    x = resample(x, sr/g, srIn/g);
end

if doDenoise
    x = sa_denoise(x, sr);
end

nyq = sr/2;
lo = max(cfg.band(1), 1) / nyq;
hi = min(cfg.band(2), 0.99*nyq) / nyq;
if lo < hi
    [z,p,k] = butter(4, [lo hi], 'bandpass');
    [sos,g] = zp2sos(z,p,k);
    x = filtfilt(sos, g, x);
end

x = x ./ (max(abs(x)) + 1e-12);

n = round(sr * cfg.clip_s);
if numel(x) >= n
    s = floor((numel(x)-n)/2) + 1;
    y = x(s:s+n-1);
else
    y = [x; zeros(n-numel(x),1)];
end
end

function y = sa_denoise(x, sr) %#ok<INUSD>
% Spectral subtraction; noise floor taken from the quietest 10% of frames.
% Framing, padding and overlap-add replicate scipy.signal.stft/istft exactly
% (periodic Hann, nfft/2 zero padding at both ends, end padding to a whole
% number of hops, window-squared normalisation), so MATLAB and Python denoise
% to the same samples. MATLAB's own stft/istft pad differently and the two
% front ends would otherwise disagree on every denoised feature.
nfft = 1024; hop = 256;
x = x(:);
if numel(x) < 2*nfft, y = x; return; end
w  = 0.5 - 0.5*cos(2*pi*(0:nfft-1)'/nfft);              % periodic Hann
xp = [zeros(nfft/2,1); x; zeros(nfft/2,1)];
nadd = mod(mod(-(numel(xp)-nfft), hop), nfft);
xp = [xp; zeros(nadd,1)];
nfr = 1 + (numel(xp)-nfft)/hop;
idx = (1:nfft)' + hop*(0:nfr-1);
F   = fft(xp(idx) .* w);
F   = F(1:nfft/2+1, :);
mag = abs(F); ph = angle(F);
e = sum(mag, 1);
k = max(1, floor(0.10*numel(e)));                        % int() in Python truncates
[~, ord] = sort(e);
noise = mean(mag(:, ord(1:k)), 2);
clean = max(mag - 1.5*noise, 0.05*mag);
Z  = clean .* exp(1j*ph);
fr = real(ifft([Z; conj(Z(end-1:-1:2, :))])) .* w;
out = zeros(numel(xp),1); nrm = zeros(numel(xp),1);
for i = 1:nfr
    s = (i-1)*hop;
    out(s+1:s+nfft) = out(s+1:s+nfft) + fr(:,i);
    nrm(s+1:s+nfft) = nrm(s+1:s+nfft) + w.^2;
end
out = out(nfft/2+1:end-nfft/2);
nrm = nrm(nfft/2+1:end-nfft/2);
nz = nrm > 1e-10;
out(nz) = out(nz) ./ nrm(nz);
if numel(out) >= numel(x), y = out(1:numel(x)); else, y = [out; zeros(numel(x)-numel(out),1)]; end
end
