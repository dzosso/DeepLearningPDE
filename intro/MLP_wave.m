clear all;
close all;
clc;

%% problem parameters
% domain: [0,1] x [0,1] (t,x)
initialpts = 50;
boundarypts = 50;
c = 0.25;
initialfct = @(x) sin(5*pi*x).^2.*(x<0.4).*(x>0.2);
solution = @(t,x) initialfct(x - c*t);
res = 100;

%% setting up the layers

layers = [ sequenceInputLayer( 2, 'Name', 'inputLayer' )
    fullyConnectedLayer(50, 'Name', 'fc1')
    tanhLayer('Name', 'hidden1' );
    fullyConnectedLayer(50, 'Name', 'fc2')
    tanhLayer('Name', 'hidden2')
    fullyConnectedLayer(10, 'Name', 'fc3')
    tanhLayer('Name', 'hidden3')
    fullyConnectedLayer(1, 'Name', 'fco')
    ];

lgraph = layerGraph(layers);

dlnet = dlnetwork(lgraph);

%% optimization variables
averageGrad = [];
averageSqGrad = [];
velocity = [];
numEpochs = 100;
miniBatchSize = 250;
numObservations = 10*miniBatchSize;
numIterationsPerEpoch = floor(numObservations./miniBatchSize);
initialLearnRate = 0.000001;
momentum = 0.5;
decay = 0.0001;
losses = [];

iteration = 0;
start = tic;
tx = linspace(0,1,res);
[tt,xx] = meshgrid(tx);
dlx = dlarray(single([tt(:)'; xx(:)']), 'CBT');

%% actual optimization
for epoch = 1:numEpochs
    for i = 1:numIterationsPerEpoch
        iteration = iteration + 1;
        
        X = rand(2,miniBatchSize); % interior points
        Xinit = [zeros(1,initialpts); rand(1,initialpts)]; % t = 0
        bpts = rand(1,boundarypts);
        Xboundary = [bpts bpts; zeros(1,boundarypts) ones(1,boundarypts)]; % bdry
        
        dlX = dlarray(single([X Xinit Xboundary]), 'CBT');
        
        % compute gradients and update parameters
        [gradients,loss] = dlfeval(@(net,X) modelGradients(net,X,miniBatchSize,initialpts,boundarypts,c,initialfct),dlnet,dlX);
        [dlnet.Learnables,averageGrad,averageSqGrad] = adamupdate(dlnet.Learnables,gradients,averageGrad,averageSqGrad,iteration);
        
        %learnRate = initialLearnRate/(1 + decay*iteration);
        %[dlnet.Learnables, velocity] = sgdmupdate(dlnet.Learnables, gradients, velocity, learnRate, momentum);
        %[dlnet.Learnables,averageSqGrad] = rmspropupdate(dlnet.Learnables,gradients,averageSqGrad);
    end
    
    % for visualization: show intermediate results
    y = forward(dlnet,dlx);
    u = reshape(double(gather(extractdata(y))), res,res)';
    losses(end+1) = double(gather(extractdata(loss)));
      
    subplot(221);
    imagesc( u ); title("trained network"); 
    xlabel('x'); ylabel('t'); daspect([1 1 1]); set(gca,"YDir", "normal", "CLim", [0,1]);
    
    subplot(222);
    imagesc( solution(tt,xx)' ); title("known solution"); 
    xlabel('x'); ylabel('t'); daspect([1 1 1]); set(gca,"YDir", "normal", "CLim", [0,1]);
       
    subplot(223);
    semilogy(losses); title("loss");
    xlabel("epoch");
    
    subplot(224);
    XX = extractdata(dlX);
    scatter(XX(2,:),XX(1,:)); title("MiniBatch of points");
    set(gca,"XLim", [0,1],"YLim", [0,1]);
    xlabel('x'); ylabel('t'); daspect([1 1 1]);  
    
    drawnow;
end


%% how we compute the loss and its gradient
function [gradients,loss] = modelGradients(dlnet,dlX,miniBatchSize,initialpts,boundarypts,c,initialfct)
    
    u = forward(dlnet,dlX);
    
    % automatic differentiation for u_t and u_x !!
    nablau = dlgradient( sum(u,"all"), dlX );
    
    operatorterms = sum((nablau(1,:) + c*nablau(2,:)).^2,"all");
    initialterms = sum((u(miniBatchSize+(1:initialpts))-initialfct(dlX(2,miniBatchSize+(1:initialpts)))).^2, "all");
    boundaryterms = sum((u(miniBatchSize+initialpts+(1:boundarypts))-u(miniBatchSize+initialpts+boundarypts+(1:boundarypts))).^2, "all"); 
    
        %+ sum((nablau(2,miniBatchSize+initialpts+(1:boundarypts))-nablau(2,miniBatchSize+initialpts+boundarypts+(1:boundarypts))).^2, "all");
    
    loss = 100*operatorterms + 500*initialterms + 200*boundaryterms;
    
    % automatic differentiation for parameter gradients
    gradients = dlgradient(loss,dlnet.Learnables);
    
end
