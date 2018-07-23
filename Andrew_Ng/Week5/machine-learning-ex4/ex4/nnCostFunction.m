function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a three layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network

% nn_params(:)

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables

m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% ================================ COST  =========================================

% size(Theta2);

% Theta2

% theta1sum
% theta2sum

% th1 = Theta1.^2;
% th2 = Theta2.^2;

% theta1_sum = sum(th1(:))
% theta2_sum = sum(th2(:))

temp_y = zeros(size(Theta2,1),1);
a1 = [ones(m,1) X];
L = [input_layer_size, hidden_layer_size, num_labels];
theta1sum=0;
theta2sum=0;
cost=0;	

for i=1:m
	a2 = size(Theta1,1);
	a2 = Theta1*a1(i,:)';
	a2 = sigmoid(a2);
	
	a2 = [ones(1,1);a2];
	a3 = size(Theta2,1);
	a3 = Theta2*a2;
	hf = sigmoid(a3);
	
	for j=1:num_labels
		if(j==y(i)) temp_y(j)=1;
		else temp_y(j)=0;
		end
		cost = cost + temp_y(j)*log(hf(j))+(1-temp_y(j))*log(1-hf(j));
	end
	
%============================= Backprop ====================================		
	
	delta3 = size(a3);
	delta3 = hf-temp_y;
	
	delta2 = size(a2);
	delta2 = (Theta2'*delta3).*(a2.*(1-a2));	
	% size(delta2)
	
	Theta1_grad = Theta1_grad + delta2(2:end)*a1(i,:);
	Theta2_grad = Theta2_grad + delta3*a2';
	
	
	%{
	
	TRIED UNVECTORISED SOLUTION BUT FAILED!!!!
	
	for i2=1:size(L,2)-1
		for j2=1:L(1,i2+1)
			for k2=1:L(1,i2)+1
				if(i2==1)
					Theta1_grad(j2,k2) = Theta1_grad(j2,k2) + a1(k2)*delta2(j2);
				elseif(i2==2)
					Theta2_grad(j2,k2) = Theta2_grad(j2,k2) + a2(k2)*delta3(j2);
				end
			end
		end
	end
	
	%}
	

					
end

for i1=1:size(L,2)-1
	for j1=1:L(1,i1+1)
		for k1=2:L(1,i1)+1
			if(i1==1)
				theta1sum = theta1sum + Theta1(j1,k1)*Theta1(j1,k1);
			elseif(i1==2)
				theta2sum = theta2sum + Theta2(j1,k1)*Theta2(j1,k1);
			end
		end
	end
end


J=J+(-1/m)*cost + (lambda/(2*m))*(theta1sum + theta2sum);



for i=1:size(L,2)-1
	for j=1:L(1,i+1)
		for k=1:L(1,i)+1
			if(i==1)
				if(k==1)
					Theta1_grad(j,k) = (1/m)*Theta1_grad(j,k);
				elseif(k!=1)
					Theta1_grad(j,k) = (1/m)*Theta1_grad(j,k) + (lambda/m)*Theta1(j,k);	
				end
			elseif(i==2)
				if(k==1)   
					Theta2_grad(j,k) = (1/m)*Theta2_grad(j,k);
				elseif(k!=1)
					Theta2_grad(j,k) = (1/m)*Theta2_grad(j,k) + (lambda/m)*Theta2(j,k);	
				end
			end
		end
	end
end





% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
