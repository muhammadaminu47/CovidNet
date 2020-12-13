%%% Clear workspace
clear; close all; clc;
 
%%% Images Datapath 
datapath = 'CT Data 3';

%%% Image Datastore
imds = imageDatastore(datapath,'IncludeSubfolders',true,'LabelSource','foldernames');

%%% Number of Images
numImages = numel(imds.Labels);

%%% Number of categories
numClasses = numel(categories(imds.Labels));

%%% Determine the number of Observations in each Class
total_split = countEachLabel(imds)

%%% Visualize random images
perm=randperm(numImages,6);
figure;
for idx=1:length(perm)
    
    subplot(2,3,idx);
    imshow(imread(imds.Files{perm(idx)}));
    title(sprintf('%s',imds.Labels(perm(idx))))
    
end

%%% Number of splits
num_splits=1; % modify this according to your required no. of splits

%%% Initialize Variables to Store Performance Information
Accuracy = zeros(num_splits,1);
Precision = zeros(num_splits,1);
Sensitivity = zeros(num_splits,1);
Specificity = zeros(num_splits,1);
F1_Score = zeros(num_splits,1);
AUC = zeros(num_splits,1);

%%% Loop for each split
for split_idx = 1:num_splits
    
fprintf('Processing %d of %d splits \n',split_idx,num_splits);

%%% Split Data into Training and Validation Sets
[imdsTrain,imdsTest] = splitEachLabel(imds,0.8,'randomized');
numTrainImages = numel(imdsTrain.Labels);

%%% Load CovidNet
load CovidNet
net = CovidNet;

%%% Pre-process Images For CovidNet
inputSize = net.Layers(1).InputSize;

augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, 'ColorPreprocessing', 'rgb2gray');
augimdsTest = augmentedImageDatastore(inputSize(1:2),imdsTest, 'ColorPreprocessing', 'rgb2gray');

%%% Extract feature vectors
layer = 'fc_1';
featuresTrain = activations(net,augimdsTrain,layer,'OutputAs','rows');
featuresTest = activations(net,augimdsTest,layer,'OutputAs','rows');

YTrain = imdsTrain.Labels;
YTest = imdsTest.Labels;

%%% Train classifier
% t = templateSVM('Standardize',false,'KernelFunction','gaussian');
opts = struct('Optimizer','bayesopt','ShowPlots',true,...
    'AcquisitionFunctionName','expected-improvement-plus');
% classifier = fitcecoc(featuresTrain,YTrain,'Learners', t, 'Coding', 'onevsall', 'ObservationsIn', 'rows',...
%     'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',opts);
% classifier = fitcsvm(featuresTrain,YTrain,'Standardize',true,'KernelFunction','rbf',...
%     'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',opts);
classifier = fitcknn(featuresTrain,YTrain,'NSMethod','exhaustive',...
    'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',opts);
% classifier = fitcensemble(featuresTrain,YTrain,'Method','Bag',...
%     'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',opts);
% classifier = fitctree(featuresTrain,YTrain,...
%     'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',opts);

%%% Classify Validation Images and Evaluate Classifier Performance
YPred = predict(classifier,featuresTest);
accuracy = mean(YPred == YTest)
cp = classperf(double(nominal(YTest)),double(nominal(YPred)));
[FPR,TPR,T,AUC] = perfcurve(double(nominal(YTest)),double(nominal(YPred)),2);

%%% Store Performace Statistics
Accuracy(split_idx) = cp.CorrectRate;
Precision(split_idx) = cp.PositivePredictiveValue;
Sensitivity(split_idx) = cp.Sensitivity;
Specificity(split_idx) = cp.Specificity;
F1_Score(split_idx) = 2*cp.PositivePredictiveValue*cp.Sensitivity./(cp.PositivePredictiveValue+cp.Sensitivity);
AUC(split_idx) = AUC;

%%% Plot Confusion Matrices
figure;
plotconfusion(YTest,YPred)
xlabel('True Label')
ylabel('Predicted Label')
title('CovidNet')

%%% Plot ROC curve
figure;
plot(FPR,TPR,'linewidth',2)
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC curve')

end

% %%% Get the CovidNet weights for the first convolutional layer
w1 = net.Layers(2).Weights;

%%% Scale and resize the weights for visualization
w1 = mat2gray(w1);
w1 = imresize(w1,5); 

%%% Display a montage of network weights.
figure;
montage(w1,'Size', [2 4])
title('First convolutional layer weights')

% %%% Visualize Features from Deeper Layer Weights
channels = 1:16;
I = deepDreamImage(net,layer,channels, ...
    'PyramidLevels',1, ...
    'Verbose',0);
    
figure
I = imtile(I,'ThumbnailSize',[250 250]);
imshow(I)
title(['Layer ',layer,' Features'],'Interpreter','none')
    
