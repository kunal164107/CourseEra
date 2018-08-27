function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

cost = 0;
Theta_reg = 0;
X_reg = 0;

	for i = 1:size(R,1)
		for j = 1:size(R,2)
			if(R(i,j)==1)		
				cost = cost + (Theta(j,:)*X(i,:)' - Y(i,j))^2;
	
			end	
		end	
	end
	
	for j = 1:size(R,2)
		for k=1:size(X,2)
			Theta_reg = Theta_reg + Theta(j,k)^2;
		end
	end
	
	for i = 1:size(R,1)
		for k=1:size(X,2)
			X_reg = X_reg + X(i,k)^2;
		end
	end
	
	
	
	 J = (1/2)*cost + (lambda/2)*(Theta_reg+X_reg);

Theta_grad1 = zeros(size(Theta_grad));
X_grad1 = zeros(size(X_grad));

	for i = 1:size(R,1)
		for j = 1:size(R,2)
			for k=1:size(X,2)
				if(R(i,j)==1)
					Theta_grad1(j,k) = Theta_grad1(j,k) + (Theta(j,:)*X(i,:)' - Y(i,j))*X(i,k);
				end
			end
		end		
	end
	
	for j = 1:size(R,2)
		for k=1:size(X,2)
			Theta_grad(j,k) = Theta_grad1(j,k) +  lambda*Theta(j,k);
		end
	end	
	
	

	for i = 1:size(R,1)
		for j = 1:size(R,2)
			for k=1:size(X,2)
				if(R(i,j)==1)
					X_grad1(i,k) = X_grad1(i,k) + (Theta(j,:)*X(i,:)' - Y(i,j))*Theta(j,k);
				end
			end
		end		
	end	


	for i = 1:size(R,1)
		for k=1:size(X,2)
			X_grad(i,k) = X_grad1(i,k) +  lambda*X(i,k);
		end
	end	



% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
