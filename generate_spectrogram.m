% Load RF dataset
data = load('rf_dataset.mat');
dataset = data.dataset;
labels = data.labels;
fs = data.fs;

% Create folder for images
outputFolder = 'Spectrograms';
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

% Convert signals to spectrograms and save as images
for i = 1:100  % ✅ Save first 100 samples (Adjust if needed)
    signal = dataset(i, :);
    snrLabel = labels(i);

    % Create spectrogram
    figure('Visible', 'off');  % Hide figure window
    spectrogram(signal, 256, 200, 512, fs, 'yaxis');
    colormap jet;
    colorbar off;
    
    % Save image
    filename = fullfile(outputFolder, sprintf('SNR_%d_Sample_%d.png', snrLabel, i));
    saveas(gcf, filename);
    close;
end

disp("✅ Spectrogram images saved successfully!");
