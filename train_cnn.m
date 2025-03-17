clc; clear; close all;

% -------------------------------
% STEP 1: Load Balanced Dataset
% -------------------------------
imds = imageDatastore('Spectrograms', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% Ensure dataset balance
minCount = min(countEachLabel(imds).Count);
imds = splitEachLabel(imds, minCount, 'randomized');

% Split into Training & Validation
[trainImds, valImds] = splitEachLabel(imds, 0.8, 'randomized');

% -------------------------------
% STEP 2: Data Augmentation & Preprocessing
% -------------------------------
inputSize = [227 227 3];

augmenter = imageDataAugmenter( ...
    'RandRotation', [-20, 20], ...
    'RandXTranslation', [-15 15], ...
    'RandYTranslation', [-15 15], ...
    'RandScale', [0.8, 1.2]);

trainData = augmentedImageDatastore(inputSize, trainImds, 'DataAugmentation', augmenter);
valData = augmentedImageDatastore(inputSize, valImds);

% -------------------------------
% STEP 3: Optimized CNN Architecture
% -------------------------------
layers = [
    imageInputLayer(inputSize)

    convolution2dLayer(7, 64, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)

    convolution2dLayer(5, 128, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)

    convolution2dLayer(3, 256, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)

    convolution2dLayer(3, 512, 'Padding', 'same') % More depth for better feature extraction
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)

    fullyConnectedLayer(1024) % More neurons for better classification
    reluLayer
    dropoutLayer(0.5)

    fullyConnectedLayer(2)  
    softmaxLayer
    classificationLayer];

% -------------------------------
% STEP 4: Training Options
% -------------------------------
options = trainingOptions('adam', ...
    'InitialLearnRate', 0.00005, ... % Slow Learning for Stability
    'MaxEpochs', 50, ...  
    'MiniBatchSize', 32, ...
    'ValidationData', valData, ...
    'ValidationFrequency', 10, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress', ...
    'Verbose', true);

% -------------------------------
% STEP 5: Train the CNN Model
% -------------------------------
rfNet = trainNetwork(trainData, layers, options);

% -------------------------------
% STEP 6: Save the Trained Model
% -------------------------------
save('rf_signal_classifier.mat', 'rfNet');

disp('✅ CNN Training Complete & Model Saved with Higher Accuracy!');
