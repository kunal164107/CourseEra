function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
% ====================== UNVECTORIZED SOLUTION ======================
	% theta1=0;
	% theta2=0;
	% theta3=0;
	
	% for i=1:m
		% for j=1:2
			% theta1 = theta1+theta(j,1)*X(i,j); 
		% end
		% theta2 = theta2 + (theta1-y(i,1))*X(i,1);
		% theta3 = theta3 + (theta1-y(i,1))*X(i,2);
		% theta1=0;
	% end
	
	% temp_theta1 = theta(1,1) - (alpha/m)*theta2;
	% temp_theta2 = theta(2,1) - (alpha/m)*theta3;
	
	% theta(1,1) = temp_theta1;
	% theta(2,1) = temp_theta2;

% ============================================================

	
% ====================== VECTORIZED SOLUTION ======================
	
	h1 = X*theta;
	h2 = (h1-y).*X(:,1);
	h3 = (h1-y).*X(:,2);
	h4 = sum(h2);
	h5 = sum(h3);
	temp_theta1 = theta(1,1) - (alpha/m)*h4;
	temp_theta2 = theta(2,1) - (alpha/m)*h5;
	
	theta(1,1) = temp_theta1;
	theta(2,1) = temp_theta2;

% ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
