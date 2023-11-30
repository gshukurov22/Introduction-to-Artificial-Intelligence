% tasks 1-5
% Load data from text file
sunspot = load('sunspot.txt');

% Create figure and plot the data
figure(1);
plot(sunspot(:, 1), sunspot(:, 2), 'r-*');

% Add title, labels, grid lines
title('Sunspot Data Plot');
xlabel('Year');
ylabel('Sun plums');
grid on;

% Set input data and output data vector (order n=2)
L = length(sunspot); % data length
P = [sunspot(1:L-11,2)' ; % input data matrix
     sunspot(2:L-10,2)'
     sunspot(3:L-9,2)'
     sunspot(4:L-8,2)'
     sunspot(5:L-7,2)'
     sunspot(6:L-6,2)'
     sunspot(7:L-5,2)'
     sunspot(8:L-4,2)'
     sunspot(9:L-3,2)'
     sunspot(10:L-2,2)'
     sunspot(11:L-1,2)'];   
T = sunspot(12:L,2)'; % output data vector
disp('Size of P:')
disp(size(P));
disp('Size of T:')
disp(size(T));
fprintf('\n');


% task 6
figure(2);
plot3(P(1,:),P(2,:),T,'bo');

% Add title, labels, grid lines
title('Sunspot 3D Data Plot');
xlabel('SP');
ylabel('SP');
zlabel('SP');
grid on;


% task 7
% Set training data sets
Pu = P(:,1:200);
Tu = T(:,1:200);


            % Pu, Tu
% task 16
% Create direct neuron
net = newlin(Pu, Tu, 0, 0.00000020469098638907);

% task 17
% Set the goal and number of epochs
net.trainParam.goal = 100;
net.trainParam.epochs = 10000;

% task 18
% Train and simulate the network
net = train(net,Pu,Tu);
Tsu = sim(net,Pu);

% Assign weights to auxiliary variables
w1 = net.IW{1}(1);
w2 = net.IW{1}(2);
w3 = net.IW{1}(3);
w4 = net.IW{1}(4);
w5 = net.IW{1}(5);
w6 = net.IW{1}(6);
w7 = net.IW{1}(7);
w8 = net.IW{1}(8);
w9 = net.IW{1}(9);
w10 = net.IW{1}(10);
w1111 = net.IW{1}(11);
b = net.b{1};

% Display weights
disp('Training data')
disp('Weight coefficient values:')
disp(net.IW{1})
disp(net.b{1})

maxlr = maxlinlr(Pu, b);

e = Tu-Tsu;
mse_value = mse(e);

% Error histogram
figure(3);
histogram(e, 20);
title('Forecast error histogram (newlin)');
xlabel('Error');
ylabel('Frequency');

fprintf('MSE (Pu, Tu): %.4f\n', mse_value);
fprintf('MaxlinLr: %.20f\n', maxlr);
fprintf('\n');

            % P, T
% task 16
% Create direct neuron
net2 = newlin(P, T, 0, 0.00000010346413313300);

% task 17
% Set the goal and number of epochs
net2.trainParam.goal = 100;
net2.trainParam.epochs = 10000;

% task 18
% Train and simulate the network
net2 = train(net2,P,T);
Ts = sim(net2,P);

% Assign weights to auxiliary variables
w11 = net2.IW{1}(1);
w22 = net2.IW{1}(2);
w33 = net2.IW{1}(3);
w44 = net2.IW{1}(4);
w55 = net2.IW{1}(5);
w66 = net2.IW{1}(6);
w77 = net2.IW{1}(7);
w88 = net2.IW{1}(8);
w99 = net2.IW{1}(9);
w100 = net2.IW{1}(10);
w111 = net2.IW{1}(11);
b2 = net2.b{1};

% Display weights
disp('All data')
disp('Weight coefficient values:')
disp(net2.IW{1})
disp(net2.b{1})

maxlr2 = maxlinlr(P, b2);

e2 = T-Ts;
mse_value2 = mse(e2);

fprintf('MSE (P, T): %.4f\n', mse_value2);
fprintf('MaxlinLr: %.20f\n', maxlr2);  
