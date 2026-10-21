function [label, conf] = sa_demo(wavFile, domain, outPng)
% MATLAB end-to-end demo: read -> M2 pre-process -> M3 features -> classify
% with the model exported from Python -> 4-panel figure.
%
%   addpath('matlab');
%   sa_demo('data/demo_wavs/heart_abnormal_01.wav', 'heart');
%   sa_demo('data/demo_wavs/music_jazz_01.wav', 'music', 'results/figures/matlab_demo.png');
if nargin < 3, outPng = ''; end
cfg = sa_config(domain);
[x, srIn] = audioread(wavFile);
x = mean(x, 2);
[y, sr] = sa_preprocess(x, srIn, cfg);

% Classify every clip-length segment and average the votes - the same rule
% the Python app uses, since one 5 s window is a fragile sample of a song.
g = gcd(round(srIn), cfg.sr);
full = resample(x, cfg.sr/g, srIn/g);
n = round(cfg.sr * cfg.clip_s);
k = max(1, floor(numel(full) / n));
scores = [];
for i = 1:k
    seg = full((i-1)*n+1 : min(i*n, numel(full)));
    [ys, srs] = sa_preprocess(seg, cfg.sr, cfg);
    [~, ~, sc] = sa_classify(sa_features(ys, srs, cfg), domain);
    if isempty(scores), scores = sc; else, scores.Score = scores.Score + sc.Score; end
end
scores.Score = scores.Score / k;
[conf, bi] = max(scores.Score);
label = char(scores.Class(bi));
fprintf('%s  ->  %s  (k-NN vote %.0f%%, %d segments)\n', wavFile, label, 100*conf, k);
disp(scores);

if isempty(outPng) && ~usejava('desktop'), return; end
f = figure('Color','w','Position',[100 100 1100 700], 'Visible', 'off');
try, theme(f, 'light'); catch, end   % R2025a+ defaults to the desktop (dark) theme
t = (0:numel(y)-1)/sr;
fmax = min(cfg.band(2)*1.15, sr/2);
subplot(2,2,1); plot(t, y); grid on;
title('Pre-processed waveform'); xlabel('Time (s)');
subplot(2,2,2);
Y = abs(fft(y)); fr = (0:numel(y)-1)*sr/numel(y); h = 1:floor(numel(y)/2);
semilogy(fr(h), Y(h)+1e-9); xlim([0 fmax]); grid on;
title('FFT magnitude spectrum'); xlabel('Hz');
subplot(2,2,3);
nwin = sa_round(sr*cfg.frame_ms/1000);
spectrogram(y, hamming(nwin), nwin-sa_round(sr*cfg.hop_ms/1000), [], sr, 'yaxis');
ylim([0 fmax/1000]); title('Spectrogram');
subplot(2,2,4);
bar(categorical(scores.Class), scores.Score); ylim([0 1]); grid on;
title(sprintf('Prediction: %s (%.0f%%)', upper(label), 100*conf));
sgtitle(sprintf('MATLAB pipeline - %s - %s', domain, wavFile), 'Interpreter','none');
if ~isempty(outPng), exportgraphics(f, outPng, 'Resolution', 110); end
if ~usejava('desktop'), close(f); else, set(f,'Visible','on'); end
end
