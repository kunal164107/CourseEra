function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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


% ====================== VECTORIZED SOLUTION ======================

% thetatransX = X*theta;
% hf = sigmoid(thetatransX);

% cost = -log(hf')*y - log(1-hf')*(1-y);
% theta2=theta.^2;
% theta1 = sum(theta2(2:end));

% J=J+(1/m)*cost+(lambda/(2*m))*theta1;

	
% =============================================================

% ====================== UNVECTORIZED SOLUTION ======================

cost=0;
thetatransX = X*theta;
hf = sigmoid(thetatransX);

n=length(grad);
theta1=0;

for i=1:m
	cost = cost + -y(i,1)*log(hf(i,1)) - (1-y(i,1))*log(1-hf(i,1));
end

for j=2:n
	theta1 = theta1+theta(j,1)*theta(j,1);
end

J=J+(1/m)*cost+(lambda/(2*m))*theta1;

% =============================================================


% ====================== GRADIENT DESCENT VECTORIZED SOLUTION ======================


% grad1 = hf-y;

% grad(1,1) = (grad1'*X(:,1))/m;

% for i=2:n
	% grad(i,1) = (grad1'*X(:,i))/m + (lambda/m)*theta(i,1);
% end



% =============================================================

% ====================== GRADIENT DESCENT UNVECTORIZED SOLUTION ======================

grad1=zeros(size(theta));


for i=1:m
	for j=1:n
	grad1(j,1) = grad1(j,1) + (hf(i,1)-y(i,1))*X(i,j); 
	end
end

grad(1,1) = grad1(1,1)/m;

for i=2:n
	grad(i,1) = grad1(i,1)/m + (lambda/m)*theta(i,1);
end




% =============================================================


% =============================================================

end
