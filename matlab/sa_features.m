function [v, names] = sa_features(x, sr, cfg)
% M3 feature extraction - MATLAB mirror of src/features.py.
% Produces the same 53-element vector in the same order, so a model trained
% in Python can be applied directly to features computed here.

frameLen = sa_round(sr * cfg.frame_ms / 1000);
hopLen   = max(1, sa_round(sr * cfg.hop_ms / 1000));
nfft     = 2^nextpow2(frameLen);

% ---- framing (Hamming) and framing (rectangular, for time-domain features)
framesW = sa_frame(x, frameLen, hopLen, true);
framesR = sa_frame(x, frameLen, hopLen, false);

% ---- FFT stage
S     = fft(framesW, nfft, 2);
S     = S(:, 1:nfft/2+1);
power = (abs(S).^2) / nfft;
freqs = (0:nfft/2) * (sr/nfft);

% ---- MFCC (pre-emphasis -> frame -> window -> FFT -> mel -> log -> DCT)
xp      = [x(1); x(2:end) - 0.97*x(1:end-1)];
framesP = sa_frame(xp, frameLen, hopLen, true);
Sp      = fft(framesP, nfft, 2); Sp = Sp(:,1:nfft/2+1);
powerP  = (abs(Sp).^2) / nfft;

fb      = sa_melbank(sr, nfft, cfg.n_mels, cfg.band(1), min(cfg.band(2), sr/2));
logMel  = log(powerP * fb.' + 1e-12);
M       = dct(logMel, [], 2);
M       = M(:, 1:cfg.n_mfcc);
dM      = sa_delta(M, 2);

% ---- spectral descriptors
psum     = sum(power, 2) + 1e-12;
centroid = (power * freqs.') ./ psum;
dev      = (repmat(freqs, size(power,1), 1) - centroid).^2;
bandw    = sqrt(sum(power .* dev, 2) ./ psum);
c        = cumsum(power, 2);
[~, ri]  = max(c >= 0.85*c(:,end), [], 2);
rolloff  = freqs(ri).';
gm       = exp(mean(log(power + 1e-12), 2));
flat     = gm ./ (mean(power, 2) + 1e-12);

% ---- time-domain descriptors
zcr = mean(abs(diff(sign(framesR), 1, 2)) > 0, 2);
ste = sum(framesR.^2, 2) / frameLen;

% ---- assemble (order must match features.py exactly)
v = []; names = {};
    function push(g, nm, a)
        v(end+1) = mean(a); names{end+1} = sprintf('%s::%s_mean', g, nm); %#ok<AGROW>
        v(end+1) = std(a, 1); names{end+1} = sprintf('%s::%s_std', g, nm); %#ok<AGROW>
    end

for i = 1:cfg.n_mfcc, push('mfcc', sprintf('mfcc%d', i), M(:,i)); end
for i = 1:cfg.n_mfcc
    v(end+1) = mean(dM(:,i)); names{end+1} = sprintf('mfcc::dmfcc%d_mean', i); %#ok<AGROW>
end
push('spectral','centroid',centroid);  push('spectral','bandwidth',bandw);
push('spectral','rolloff',rolloff);    push('spectral','flatness',flat);
push('temporal','zcr',zcr);            push('temporal','energy',ste);
push('temporal','rms',sqrt(ste));

v = v(:).';
v(~isfinite(v)) = 0;
end


function F = sa_frame(x, frameLen, hopLen, useWindow)
x = x(:);
if numel(x) < frameLen, x = [x; zeros(frameLen-numel(x),1)]; end
n = 1 + floor((numel(x)-frameLen)/hopLen);
idx = (0:frameLen-1).' + (0:n-1)*hopLen + 1;
F = x(idx).';
if useWindow, F = F .* hamming(frameLen).'; end
end


function fb = sa_melbank(sr, nfft, nMels, fmin, fmax)
hz2mel = @(f) 2595*log10(1 + f/700);
mel2hz = @(m) 700*(10.^(m/2595) - 1);
mels = linspace(hz2mel(fmin), hz2mel(fmax), nMels+2);
hz   = mel2hz(mels);
bins = min(max(floor((nfft+1)*hz/sr), 0), nfft/2);
fb   = zeros(nMels, nfft/2+1);
for m = 1:nMels
    l = bins(m); c = bins(m+1); r = bins(m+2);
    if c == l, c = min(l+1, nfft/2); end
    if r == c, r = min(c+1, nfft/2); end
    if c > l, fb(m, l+1:c)   = ((l:c-1) - l) / (c - l); end
    if r > c, fb(m, c+1:r)   = (r - (c:r-1)) / (r - c); end
end
enorm = 2 ./ (hz(3:nMels+2) - hz(1:nMels) + 1e-12);
fb = fb .* enorm(:);
end


function D = sa_delta(F, w)
if size(F,1) < 3, D = zeros(size(F)); return; end
P = [repmat(F(1,:), w, 1); F; repmat(F(end,:), w, 1)];
D = zeros(size(F));
ns = -w:w;
for i = 1:numel(ns)
    D = D + ns(i) * P(i:i+size(F,1)-1, :);
end
D = D / (2 * sum((1:w).^2));
end
