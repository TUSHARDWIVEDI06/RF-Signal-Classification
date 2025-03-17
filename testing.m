% Load Test Data
testImds = imageDatastore('Spectrograms', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
inputSize = [227 227 3];  
testData = augmentedImageDatastore(inputSize, testImds);

% Classify Test Signals
predictedLabels = classify(rfNet, testData);
actualLabels = testImds.Labels;

% Evaluate Accuracy
accuracy = sum(predictedLabels == actualLabels) / numel(actualLabels);
disp(['✅ Test Accuracy: ', num2str(accuracy * 100), '%']);

% Check Predictions with Images
idx = randperm(numel(testImds.Files), 9);
figure;
for i = 1:9
    subplot(3,3,i);
    img = readimage(testImds, idx(i));
    imshow(img);
    title(['Predicted: ', char(predictedLabels(idx(i)))]);
end
