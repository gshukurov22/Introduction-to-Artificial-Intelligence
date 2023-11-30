% Load the dataset
data = readtable('normalized.csv');

% Set the target attribute
target = 'price';

% Split the data into inputs and targets
varNames = data.Properties.VariableNames;
varNames = varNames(~ismember(varNames, target)); % excluding target attribute
inputs = data(:, varNames);
inputs = table2array(inputs);
targets = table2array(data(:, target));           % target is the last column "price"

% Create the neural network
net = feedforwardnet([12,12,12,12,12,12,12,12,12,12]); % 10 hidden layers with Logarithmic sigmoid transfer activation function:
net.layers{1}.transferFcn = 'logsig';                  % use logsig activation function for hidden layer
net.layers{2}.transferFcn = 'logsig';
net.layers{3}.transferFcn = 'logsig';
net.layers{4}.transferFcn = 'logsig';
net.layers{5}.transferFcn = 'logsig';
net.layers{6}.transferFcn = 'logsig';
net.layers{7}.transferFcn = 'logsig';
net.layers{8}.transferFcn = 'logsig';
net.layers{9}.transferFcn = 'logsig';
net.layers{10}.transferFcn = 'logsig';
net.layers{11}.transferFcn = 'purelin';        % use purelin activation function for output layer
net.trainFcn = 'trainlm';                      % use Levenberg-Marquardt algorithm for training
net.trainParam.epochs = 1000;                  % set the number of training epochs
net.trainParam.lr = 0.000247100964773;         % set the learning rate


% Configure 10 fold cross-validation
cv = cvpartition(size(inputs, 1), 'KFold', 10);

mse = zeros(1, cv.NumTestSets);
accuracy = zeros(1, cv.NumTestSets);

% Train and test the neural network for each fold
for i = 1:cv.NumTestSets
    trainIdx = cv.training(i);
    testIdx = cv.test(i);
    xTrain = inputs(trainIdx,:);
    tTrain = targets(trainIdx);
    xTest = inputs(testIdx,:);
    tTest = targets(testIdx);

    % Train the neural network
    [net, tr] = train(net, xTrain', tTrain');

    % Test the neural network
    yPred = net(xTest')';

    % Compute the accuracy of the prediction model
    diff = abs(yPred - tTest);
    acc = 100 * (1 - mean(diff./tTest));
    accuracy(i) = acc;
    disp(['Fold ', num2str(i), ' accuracy: ', num2str(acc), '%']);

    % Compute mse
    mse(i) = mean((tTest - yPred).^2);
    disp(mse);
end

% Compute the mean MSE over all folds
mean_mse = mean(mse);
disp('Mean MSE:');
fprintf('%.5f\n', mean_mse);

% Compute the mean accuracy over all folds
mean_acc = mean(accuracy);
disp('Mean accuracy:');
disp([num2str(mean_acc), '%']);

% Get the most optimal learning rate
b = net.b{1};
maxlr = maxlinlr(inputs, b);
fprintf('MaxlinLr: %.15f\n', maxlr);

