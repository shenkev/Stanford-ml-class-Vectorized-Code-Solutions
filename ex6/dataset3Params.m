function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

minWrong = length(yval);
iter = 100;
for i=1:iter
    C_can = rand*30;
    sigma_can = rand*30;
    model= svmTrain(X, y, C_can, @(x1, x2) gaussianKernel(x1, x2, sigma_can));
    ypred = svmPredict(model, Xval);
    wrongs = sum(abs(ypred-yval));
    if wrongs < minWrong
       minWrong = wrongs;
       C = C_can;
       sigma = sigma_can;
    end
    fprintf("%f \t %f \t %f \n", wrongs, C_can, sigma_can);
end

fprintf("Best Parameters: \n %f \t %f \n Lowest number of wrongs: \n %f", C, sigma, minWrong);

% =========================================================================

end
