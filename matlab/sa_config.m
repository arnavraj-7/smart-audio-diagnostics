function cfg = sa_config(domain)
% Mirror of config.py DOMAINS so MATLAB and Python use identical DSP settings.
switch lower(domain)
    case 'music'
        cfg = struct('sr',22050,'band',[30 10000],'frame_ms',25,'hop_ms',10, ...
                     'n_mels',40,'n_mfcc',13,'clip_s',5);
        cfg.classes = {'classical','jazz','rock','pop','hiphop'};
    case 'heart'
        cfg = struct('sr',2000,'band',[25 400],'frame_ms',50,'hop_ms',25, ...
                     'n_mels',26,'n_mfcc',13,'clip_s',5);
        cfg.classes = {'normal','abnormal'};
    case 'respiratory'
        cfg = struct('sr',4000,'band',[80 1800],'frame_ms',50,'hop_ms',25, ...
                     'n_mels',30,'n_mfcc',13,'clip_s',3);
        cfg.classes = {'normal','wheeze','crackle'};
    otherwise
        error('sa_config:unknownDomain','Unknown domain %s', domain);
end
end
