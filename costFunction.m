function [J, grad] = costFunction(theta, input_layer, hidden_layer, num_labels, X, Y, lambda)
  m = size(X,1);
  
  theta1 = reshape(theta(1:(input_layer+1)*hidden_layer), hidden_layer, (input_layer+1));
  theta2 = reshape(theta(((input_layer+1)*hidden_layer+1):end), num_labels, (hidden_layer+1));
  
  a1 = [ones(m,1) X];
  z1 = a1*theta1';
  a2 = [ones(size(z1,1),1) sigmoid(z1)];
  z2 = a2*theta2';
  a3 = sigmoid(z2);
  y_output = eye(num_labels);
  y_output = y_output(Y, :);
  
  reg_theta1 = theta1(:, 2:end);
  reg_theta2 = theta2(:, 2:end);
  
  J = -(1.0/m)*(sum(sum(y_output.*log(a3))) + sum(sum((1-y_output).*log(1-a3)))) + (lambda/(2*m))*(sum(sum(reg_theta1.^2)) + sum(sum(reg_theta2.^2)));
  grad = 0;
  
  del3 = a3 - y_output;
  del2 = del3*theta2.*a2.*(1-a2);
  del1 = del2(:, 2:end)'*a1; %without bias
  del2 = del3'*a2;
  
  theta_grad1 = del1/m;
  theta_grad2 = del2/m;
  
  theta_grad1(:, 2:end) = theta_grad1(:, 2:end) + (lambda/m)*(theta1(:, 2:end));
  theta_grad2(:, 2:end) = theta_grad2(:, 2:end) + (lambda/m)*(theta2(:, 2:end));
  
  grad = [theta_grad1(:); theta_grad2(:)];
endfunction
