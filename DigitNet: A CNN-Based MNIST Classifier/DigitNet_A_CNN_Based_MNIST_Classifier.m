[XTrain, YTrain] = digitTrain4DArrayData;
[XTest, YTest] = digitTest4DArrayData;

layers = [
    imageInputLayer([28 28 1])
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];

options = trainingOptions('sgdm', ...
    'MaxEpochs',5, ...
    'InitialLearnRate',0.01, ...
    'Verbose',false, ...
    'Plots','training-progress', ...
    'ValidationData',{XTest, YTest}, ...
    'ValidationFrequency',30);

net = trainNetwork(XTrain, YTrain, layers, options);

YPred = classify(net, XTest);
accuracy = sum(YPred == YTest) / numel(YTest);
fprintf('Test Accuracy: %.2f%%\n', accuracy * 100);

figure;
idx = randperm(numel(YTest), 16);
for i = 1:16
    subplot(4,4,i);
    imshow(XTest(:,:,:,idx(i)));
    title(sprintf('True: %s, Pred: %s', string(YTest(idx(i))), string(YPred(idx(i)))));
end
