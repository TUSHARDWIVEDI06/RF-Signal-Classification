% Load the combined signal
data = load('combined_signal.mat');  
combinedSignal = data.combinedSignal;  
fs = data.fs;  

% Define number of samples for training and testing
numSamples = 5000;  % Total dataset size

% Generate noise variations
SNR_range = 0:5:30;  % Different SNR levels (0dB to 30dB)
dataset = [];
labels = [];

for snr = SNR_range
    for i = 1:(numSamples / length(SNR_range))
        % Add Gaussian noise to the signal
        noisySignal = awgn(combinedSignal, snr, 'measured');
        
        % Save in dataset
        dataset = [dataset; noisySignal.'];  
        labels = [labels; snr];  % Label based on SNR value
    end
end

% Save dataset for AI training
save('rf_dataset.mat', 'dataset', 'labels', 'fs');

disp("✅ RF Dataset generated successfully!");
