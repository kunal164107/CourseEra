X = sin(magic(4));
X = X(:,1:3);
[mu sigma2] = estimateGaussian(X)



% output:
% mu =
  % -0.3978779  0.3892253  -0.0080072
% sigma2 =
   % 0.27795    0.65844   0.20414
% ---------------------------------------

[epsilon F1] = selectThreshold([1 0 0 1 1]', [0.1 0.2 0.3 0.4 0.5]')
% output:
% epsilon =  0.40040
% F1 =  0.57143