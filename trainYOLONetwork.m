%% clean workspace
clear;
clc;
%% Load .mat files
load('WordClassNames.mat')
data = load('Train.mat');
trainingData = data.TrainSample;
%% img and box lable datastores
ims = datastore(fullfile('C:\Users\asa\Desktop\DONE THINGS\NEW YOLO Network\GT\TrainDataSample'),'FileExtensions', '.bmp','Type', 'image');
blds = boxLabelDatastore(trainingData(:,2:end));
%% combine images and boxLabel Datastore 
ds = combine(ims,blds);
%% Create YOLO network
% Specify the size of input image
imageSize = [128 128 3];
%% Specify the number of object classes the network has to detect.

[numClasses,col] = size(classNames);
% Define and estimate the anchor boxes.
numAnchors = 5;
anchorBoxes = estimateAnchorBoxes(ds,numAnchors)

% Specify the pretrained ResNet -50 network as the base network for YOLO v2
net = resnet50();

%% Analyze the network architecture to view all the network layers.
analyzeNetwork(net)
% Specify the network layer to be used for feature extraction.
featureLayer = 'activation_49_relu';
% Create the YOLO v2 object detection network. 
% The network is returned as a LayerGraph object.
lgraph = yolov2Layers(imageSize,numClasses,anchorBoxes,net,featureLayer);
%% Analyze the YOLO network architecture
analyzeNetwork(lgraph)
% =====================================%%
%% Configure the network training options.
%'Plots','training-progress',...
%'ExecutionEnvironment','gpu',...
checkpointPath = pwd;
options = trainingOptions('sgdm',...
          'InitialLearnRate',0.001,...
          'Verbose',true,...
          'MiniBatchSize',10,...
          'MaxEpochs',20,...
          'Shuffle','every-epoch',...
          'VerboseFrequency',5,...
          'CheckpointPath',checkpointPath);
%% Train YOLO network
[LetterDetector,info] = trainYOLOv2ObjectDetector(ds,lgraph,options);
%% Detect the bounding box
% Read a test image into the workspace.
%img = imread('detectword.bmp');
% Run the trained YOLO v2 object detector on the test image for character detection.
%[bboxes,scores] = detect(detector,img);
% Display the detection results.

%if(~isempty(bboxes))
 %   img = insertObjectAnnotation(img,'rectangle',bboxes,scores);
%end
%figure
%imshow(img)
%% training accuracy by inspecting the training loss for each iteration.
figure
plot(info.TrainingLoss)
grid on
xlabel('Number of Iterations')
ylabel('Training Loss for Each Iteration')
%% Finshed the work
fprintf('GREAT!! YOU DO IT :)')

%blds = boxLabelDatastore(BBoxes);