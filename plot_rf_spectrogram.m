clc; clear; close all;

% -------------------------------
% STEP 1: Load Combined RF Signal
% -------------------------------
load('combined_signal.mat', 'combinedSignal', 'fs');

% -------------------------------
% STEP 2: Generate Spectrogram Parameters
% -------------------------------
window = hamming(512);  % Larger window for better resolution
noverlap = 256;         % Increased overlap for smoother transitions
nfft = 1024;            % Higher FFT resolution

% Convert signal into spectrogram
[s, f, t, ps] = spectrogram(combinedSignal, window, noverlap, nfft, fs, 'yaxis');
ps_db = 10*log10(abs(ps)); % Convert power spectrum to dB scale

% Normalize for better visualization
ps_db = ps_db - max(ps_db(:));
ps_db(ps_db < -50) = -50; % Clip low values for better contrast

% -------------------------------
% STEP 3: Load Trained CNN Model
% -------------------------------
load('rf_signal_classifier.mat', 'rfNet'); % Load trained model

% CNN input size
inputSize = [227 227 3];

% **Split spectrogram into two separate plots**
midpoint = round(size(ps_db, 2) / 2);
wifi_spectrogram = ps_db(:, 1:midpoint);
bluetooth_spectrogram = ps_db(:, midpoint+1:end);

% Resize for CNN
wifi_img = imresize(repmat(mat2gray(wifi_spectrogram), 1, 1, 3), inputSize(1:2));
bluetooth_img = imresize(repmat(mat2gray(bluetooth_spectrogram), 1, 1, 3), inputSize(1:2));

% **Predict Wi-Fi Signal**
[YPred_wifi, scores_wifi] = classify(rfNet, wifi_img);
pred_wifi = char(YPred_wifi);
conf_wifi = max(scores_wifi) * 100;

% **Predict Bluetooth Signal**
[YPred_bt, scores_bt] = classify(rfNet, bluetooth_img);
pred_bt = char(YPred_bt);
conf_bt = max(scores_bt) * 100;

% -------------------------------
% STEP 4: Display Separate Spectrograms with Clear Labels
% -------------------------------
figure;

% **Wi-Fi Spectrogram**
subplot(1,2,1);
imagesc(t(1:midpoint), f/1e6, wifi_spectrogram);
axis xy;
colormap turbo;
colorbar;
xlabel('Time (ms)');
ylabel('Frequency (MHz)');
title(sprintf('Wi-Fi Signal - %s (%.2f%%)', pred_wifi, conf_wifi));
rectangle('Position', [t(1), f(1)/1e6, t(midpoint), (f(end)-f(1))/1e6], 'EdgeColor', 'cyan', 'LineWidth', 2);

% **Bluetooth Spectrogram**
subplot(1,2,2);
imagesc(t(midpoint+1:end), f/1e6, bluetooth_spectrogram);
axis xy;
colormap turbo;
colorbar;
xlabel('Time (ms)');
ylabel('Frequency (MHz)');
title(sprintf('Bluetooth Signal - %s (%.2f%%)', pred_bt, conf_bt));
rectangle('Position', [t(midpoint+1), f(1)/1e6, t(end)-t(midpoint+1), (f(end)-f(1))/1e6], 'EdgeColor', 'magenta', 'LineWidth', 2);

hold off;
disp("✅ Separate Spectrograms for Wi-Fi & Bluetooth Generated with Clear Differences!");
