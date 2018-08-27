function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

% ====================== VECTORIZED SOLUTION ======================

thetatransX = X*theta;
hf = sigmoid(thetatransX);

cost = -log(hf')*y - log(1-hf')*(1-y);

J=J+(1/m)*cost;

	
% =============================================================
	
% ====================== UNVECTORIZED SOLUTION ======================

% cost=0;
% thetatransX = X*theta
% hf = sigmoid(thetatransX);

% for i=1:m
	% cost = cost + -y(i,1)*log(hf(i,1)) - (1-y(i,1))*log(1-hf(i,1));
% end

% J=J+(1/m)*cost;

% =============================================================

% ====================== GRADIENT DESCENT VECTORIZED SOLUTION ======================


% grad1 = hf-y;

% grad(1,1) = (grad1'*X(:,1))/m;
% grad(2,1) = (grad1'*X(:,2))/m;
% grad(3,1) = (grad1'*X(:,3))/m;

% =============================================================

% ====================== GRADIENT DESCENT UNVECTORIZED SOLUTION ======================

grad1=0;
grad2=0;
grad3=0;

for i=1:m	
	grad1 = grad1 + (hf(i,1)-y(i,1))*X(i,1); 
	grad2 = grad2 + (hf(i,1)-y(i,1))*X(i,2); 
	grad3 = grad3 + (hf(i,1)-y(i,1))*X(i,3); 
end

grad(1,1) = grad1/m;
grad(2,1) = grad2/m;
grad(3,1) = grad3/m;

% ====================== GRADIENT DESCENT UNVECTORIZED SOLUTION ======================



% =============================================================

end
