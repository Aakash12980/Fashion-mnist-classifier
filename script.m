cd 'C:/users/user/matlab drive/octave/Titanic';
data = dlmread('fashion_mnist_test.csv', ',', 1, 0);
X = data(:, 2:end);
Y = data(:,1:1);

num_labels = 10;
input_layer = 784;
hidden_layer = 30;
lambda = 0.03;

theta1 = initializeWeight(input_layer, hidden_layer);
theta2 = initializeWeight(hidden_layer, num_labels);
theta = [theta1(:);theta2(:)];

[J, grad] = costFunction(theta, input_layer, hidden_layer, num_labels, X, Y, lambda);
lambda = 3;
options = optimset('MaxIter', 100);
costFunction = @(p) costFunction(p, input_layer, hidden_layer, num_labels, X, Y, lambda);

[nn_params, cost] = fmincg(costFunction, theta, options);

Theta1 = reshape(nn_params(1:(input_layer+1)*hidden_layer), hidden_layer, (input_layer+1));
Theta2 = reshape(nn_params(((input_layer+1)*hidden_layer+1):end), num_labels, (hidden_layer+1));

pred = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == Y)) * 100);