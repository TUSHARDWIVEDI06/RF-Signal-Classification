% Step 1: Load all images
imds = imageDatastore('Spectrograms', 'IncludeSubfolders', false, 'LabelSource', 'none');

% Step 2: Convert Images to Feature Vectors
features = zeros(numel(imds.Files), 100); % Adjust 100 as needed
for i = 1:numel(imds.Files)
    img = imread(imds.Files{i});
    imgGray = rgb2gray(img);
    features(i, :) = mean(imgGray(:)); % Simplified feature extraction
end

% Step 3: Apply Clustering (K-means)
numClusters = 2;  % Assuming 2 classes: Wi-Fi & Bluetooth
[idx, C] = kmeans(features, numClusters);

% Step 4: Move Files Based on Clusters
mkdir('Spectrograms/Wi-Fi');
mkdir('Spectrograms/Bluetooth');

for i = 1:numel(imds.Files)
    if idx(i) == 1
        movefile(imds.Files{i}, 'Spectrograms/Wi-Fi/')
    else
        movefile(imds.Files{i}, 'Spectrograms/Bluetooth/')
    end
end

disp('✅ Files successfully sorted into Wi-Fi and Bluetooth folders!');

