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

% Set input data and output data vector (order n=11)
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


% task 8
% Create an artificial neuron
net = newlind(Pu, Tu);


% task 9
% Display corresponding neuron weight coefficient values
disp('Neuron weight coefficient values:' )
disp(net.IW{1})
disp(net.b{1})

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
w11 = net.IW{1}(11);
b = net.b{1};


% task 10
% Simulate the network
Tsu = sim(net,Pu);

% Comparison diagram 1
figure(3);
plot(Tsu, 'r');
hold on;
plot(Tu, 'b');
title('Comparison diagram (training data)');
legend('Forecasted values', 'True values');


% task 11 (task 8-10, but with all data)
% Create an artificial neuron
net2 = newlind(P, T);

% Simulate the network
Ts = sim(net2,P);

% Comparison diagram 2
figure(4);
plot(Ts, 'r');
hold on;
plot(T, 'b');
title('Comparison diagram (all data)');
legend('Forecasted values', 'True values');


% task 12
% Create forecast error vector
e = T - Ts;

% Error diagram
figure(5);
plot(e, 'g');
title('Forecast error diagram');
xlabel('Time step');
ylabel('Error');


% task 13
% Error histogram
figure(6);
histogram(e, 20);
title('Forecast error histogram (newlind)');
xlabel('Error');
ylabel('Frequency');

% task 14
% Calculate the mean squared error
mse_value = mse(e);

% Calculate the mean absolute error
mae_value = mae(e);

% Display the mean squared error and mean absolute error
fprintf('Mean Squared Error: %.4f\n', mse_value);
fprintf('Mean Absolute Error: %.4f\n', mae_value);

