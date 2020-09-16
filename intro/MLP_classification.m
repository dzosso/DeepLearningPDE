clear variables; close all; clc;

%% setting up data 
 
sigma = 0.45;

[XTrain, YTrain] = swissroll(1e5,sigma);
[XV, YV] = swissroll(1e4,sigma);

plotroll(XTrain, YTrain); % show the training data

%% setting up two hidden layers with 10 nodes, each, all fully connected
layers = [ sequenceInputLayer( 2 ) % 2-component input
    fullyConnectedLayer(50)
    tanhLayer
    fullyConnectedLayer(30)
    tanhLayer
    fullyConnectedLayer(20)
    tanhLayer
    fullyConnectedLayer(2)		% there are two classes, so two of these nodes
    softmaxLayer				% 
    classificationLayer			% these two are needed for classification output
    ];

%% training options
options = trainingOptions('adam', ...
    'MaxEpochs',1000,...
    ...'InitialLearnRate',1e-7, ...
    ...'Momentum', 0.95,...
    'Verbose',true, ...
    'Plots','training-progress', ...
    'ValidationData', {XV,categorical(YV)} );

%% let MATLAB do the actual training
net = trainNetwork( XTrain, categorical(YTrain), layers, options );
 
%% now use the trained network to paint entire feature space
[x1,x2]=meshgrid(linspace(-15,15,1000)); % testing x
XTest = [x1(:)'; x2(:)'];
y = net.classify(XTest); % network's prediction

YTest = zeros(1,size(XTest,2));  % convert categorical to numerical output
YTest(y == '1') = 1;
YTest(y == '-1') = -1;
 
figure; plotroll(XTest,YTest); % plot the classificatoin landscape
