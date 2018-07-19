function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
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
    %       of the cost function (computeCostMulti) and gradient here.
    %

	% ====================== UNVECTORIZED SOLUTION ======================
	
	t=size(X,2);
	theta1=0;
	theta2=zeros(t,1);
	temp_theta=zeros(t,1);
	for i=1:m
		for j=1:t
			theta1=theta1+theta(j,1)*X(i,j);
		end
		for k=1:t
			theta2(k,1) = theta2(k,1)+(theta1-y(i,1))*X(i,k);
		end;
		theta1=0;
	end
	
	for l=1:t
		temp_theta(l,1) = theta(l,1)-((alpha)*theta2(l,1))/m;
	end

	for p=1:t
		theta(p,1)=temp_theta(p,1);
	end

	% ============================================================

	% ====================== VECTORIZED SOLUTION ======================
	
	
	
	
	
	% ============================================================
	
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
