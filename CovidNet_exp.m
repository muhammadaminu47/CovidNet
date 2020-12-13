% Clear workspace
clear; close all; clc;
 
% Images Datapath 
datapath = 'Data';
 
% Image Datastore
imds = imageDatastore(datapath, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

% Number of Images
numImages = numel(imds.Labels);

% Number of categories
numClasses = numel(categories(imds.Labels));

% Determine the number of Observations in each Class
total_split = countEachLabel(imds)

% Visualize random images
perm=randperm(numImages,6);
figure;
for idx=1:length(perm)
    
    subplot(2,3,idx);
    imshow(imread(imds.Files{perm(idx)}));
    title(sprintf('%s',imds.Labels(perm(idx))))
    
end

% Number of splits
num_splits=1; % modify this according to your required no. of splits

% Initialize Variables to Store Performance Information
Accuracy = zeros(num_splits,1);
Precision = zeros(num_splits,1);
Sensitivity = zeros(num_splits,1);
Specificity = zeros(num_splits,1);
F1_Score = zeros(num_splits,1);

% Loop for each split
for split_idx = 1:num_splits
    
    fprintf('Processing %d among %d splits \n',split_idx,num_splits);
    
    % Split Data into Training and Validation Sets
    [imdsTrain,imdsTest] = splitEachLabel(imds,0.8,'randomized');
 
    % CovidNet Architecture 
    layers = [
    imageInputLayer([448 448 1])
    
    convolution2dLayer(7,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,64,'Padding','same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,128,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,256,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,512,'Padding','same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,1024,'Padding','same')
    batchNormalizationLayer
    reluLayer

    globalAveragePooling2dLayer
    fullyConnectedLayer(512)
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(numClasses)
    softmaxLayer
    
    classificationLayer];    
    
    % Pre-process Images For CovidNet
    inputSize = layers(1).InputSize;
    augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, 'ColorPreprocessing', 'rgb2gray');
    augimdsTest = augmentedImageDatastore(inputSize(1:2),imdsTest, 'ColorPreprocessing', 'rgb2gray');
    
    % Training Options
    options = trainingOptions('adam', ...
        'InitialLearnRate',0.0001, ...
        'MaxEpochs',5, ...
        'Shuffle','every-epoch', ...
        'ValidationData',augimdsTest, ...
        'ValidationFrequency',35, ...
        'Verbose',false, ...
        'Plots','training-progress');
        
    % Train CovidNet Using Training Data
    CovidNet = trainNetwork(augimdsTrain,layers,options);
    
    % Classify Validation Images and Evaluate Classifier Performance
    YPred = classify(CovidNet,augimdsTest);
    YValidation = imdsTest.Labels;
    cp = classperf(double(nominal(YValidation)),double(nominal(YPred)));
    
    % Store Performace Statistics
    Accuracy(split_idx) = cp.CorrectRate;
    Precision(split_idx) = cp.PositivePredictiveValue;
    Sensitivity(split_idx) = cp.Sensitivity;
    Specificity(split_idx) = cp.Specificity;
    F1_Score(split_idx) = 2*cp.PositivePredictiveValue*cp.Sensitivity./(cp.PositivePredictiveValue+cp.Sensitivity);
    
    % Plot Confusion Matrices
    figure;
    plotconfusion(YValidation,YPred)
    xlabel('True Label')
    ylabel('Predicted Label')
    title(['Split ',num2str(num_splits)])
    
    % Save the Independent CovidNet Architectures obtained for each split
    save(sprintf('CovidNet_%d_among_%d_splits',split_idx,num_splits),'CovidNet');
        
end


% Investigate CovidNet Predictions Using Occlusion
classes = CovidNet.Layers(end).Classes;
ind = find(imdsTest.Labels == 'Covid-19');
idz = datasample(ind,1);
img = imread(imdsTest.Files{idz});

if  size(img,3)==3
    img1 = rgb2gray(img);
    img1=imresize(img1,inputSize(1:2));
else
    img1=imresize(img,inputSize(1:2));
end

[label,scores] = classify(CovidNet,img1);
[~,topIdx] = maxk(scores, 3);
topScores = scores(topIdx);
topClasses = classes(topIdx);

figure;
imshow(img1);
titleString = compose("%s (%.2f)",topClasses,topScores');
title(sprintf(join(titleString, "; ")));

map = occlusionSensitivity(CovidNet,img1,label,...
      "Stride", 10, ...
      "MaskSize", 15);

figure;
imshow(img1,'InitialMagnification', 150)
hold on
imagesc(map,'AlphaData',0.5)
colormap jet
colorbar

title(sprintf("Occlusion sensitivity (%s)",label))

