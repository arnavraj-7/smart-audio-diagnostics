% ================================================================
% DROP-IN REPLACEMENT for the ClassifySignal method in
% SmartAudioDiagnostics.m
%
% The current version uses hard-coded spectral-centroid thresholds
% ("centroid < 1000 -> acoustic-like"). That is a placeholder, not a
% classifier. This version runs the same 53-feature DSP front end used by
% the Python training code and applies the model exported from it, so the
% GUI reports a genuinely trained prediction with a confidence score.
%
% Requires on the MATLAB path:  sa_config.m  sa_preprocess.m
%                               sa_features.m  sa_classify.m
% and model_export/ populated by:  python scripts/export_for_matlab.py
% ================================================================

function ClassifySignal(app, ~)

    if isempty(app.AudioData)
        uialert(app.UIFigure, 'Please select an audio file first.', 'No Audio');
        return;
    end

    switch app.SignalTypeDropDown.Value
        case 'Music'
            domain = 'music';
            app.ResultTitle.Text = 'Music Genre Classification';
        case 'Heart Sound'
            domain = 'heart';
            app.ResultTitle.Text = 'Heart Sound Screening';
        case 'Respiratory Sound'
            domain = 'respiratory';
            app.ResultTitle.Text = 'Respiratory Sound Screening';
        otherwise
            domain = 'heart';
            app.ResultTitle.Text = 'Medical Signal Screening';
    end

    try
        cfg = sa_config(domain);

        % M2 -> M3 : identical chain to the Python pipeline
        [y, sr] = sa_preprocess(app.AudioData, app.Fs, cfg);
        v = sa_features(y, sr, cfg);

        % M4 : trained model exported from Python
        [label, conf, scores] = sa_classify(v, domain);

        app.ResultLabel.Text = sprintf('%s  (confidence %.0f%%)', label, 100*conf);
        app.FeatureTable.Data = scores;

        if strcmp(domain, 'music')
            app.StatusLabel.Text = "Genre classified.";
        else
            app.StatusLabel.Text = ...
                "Screening result only - not a clinical diagnosis.";
        end

    catch ME
        uialert(app.UIFigure, ME.message, 'Classification Error');
    end

end
