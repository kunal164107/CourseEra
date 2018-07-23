function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% ====================== VECTORIZED SOLUTION ======================
% h2 = h1-y;
% h3 = h2.^2;
% h4 = sum(h3);
% h5 = (1/(2*m))*h4;
% J=h5;

% =============================================================

% ====================== UNVECTORIZED SOLUTION ======================
	
theta1 = 0;
theta2 = 0;
	
	for i=1:m
		for j=1:2
			theta1 = theta1+theta(j,1)*X(i,j);
		end
			theta2 = theta2+(theta1-y(i,1))^2;
			theta1=0;
	end
	
	J = (1/(2*m))*theta2;

% =========================================================================

end
