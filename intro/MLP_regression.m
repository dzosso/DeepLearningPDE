clear variables; close all; clc;

%% setting up two hidden layers with 10 nodes, each, all fully connected
layers = [ sequenceInputLayer( 1 )
    fullyConnectedLayer(10)
    tanhLayer
    fullyConnectedLayer(10)
    tanhLayer
    fullyConnectedLayer(1)
    regressionLayer
    ];

%% training data
XTrain = 2*pi*rand(1,1e4); %  x, sampled uniformly from [0,2pi]
YTrain = sin(XTrain)+0.1*randn(size(XTrain)); % corresponding y w/ noise
 
%% validation data
XV = 2*pi*rand(1,1e3);
YV = sin(XV)+0.1*randn(size(XV));

%% training options
options = trainingOptions('sgdm', ...
    'MaxEpochs',2000,...
    'InitialLearnRate',1e-5, ...
    'Verbose',true, ...
    'Plots','training-progress', ...
    'ValidationData', {XV,YV} );

%% let MATLAB do the actual training
net = trainNetwork( XTrain, YTrain, layers, options );
 
%% now use the trained network 
x = 0:0.001:(2*pi); % testing x
y = net.predict(x); % network's prediction
 
figure; 
plot( x, y ); hold on; % learned function
plot( x, sin(x) );     % true function
plot( XTrain(1:100:end), YTrain(1:100:end), 'ko'); % some data points
