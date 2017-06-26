function [ fitted_data, bounds, beta_coeffs, sigma ] = OLS(x, y, alpha)
%OLS - Ordinary Least Squares
%   Detailed explanation goes here

if nargin < 3
    alpha = 0.0027;
end

% Extract length of the data of interest
N = length(x);
% Construct X values that will be divided by Ydata.
X = zeros(N,2);
X(:,1) = 1;
X(:,2) = x;

xm = mean(x);
ym = mean(y);

% Compute intercept and slope
% beta_coeffs = flip(X \ y);

beta = sum((x - xm).*(y-ym));
beta = beta / sum( (x - xm).^2 );
alpha = ym - beta * xm;
beta_coeffs = [alpha, beta];
% Find the estimated values
fitted_data = beta_coeffs(2) * x + beta_coeffs(1);

r = fitted_data-y;
% sigma = sqrt( var(r));

SSe = sum(r.^2);         % Squared sum of errors
var_e = SSe/(N-2);       % Variance of the error
sigma = sqrt(var_e);
tval = tinv(1 - alpha/2, N-2); % Student t value with given degrees of freedom and (1-alpha) Confidence interval
c = sqrt( 1 + 1/N + (( x - xm ).^2) / sum( ( x - xm ).^2 )  ); 
bound = tval * sigma * c;

bounds = [fitted_data +  bound, fitted_data - bound ];

% bounds = [fitted_data +  1.96*sigma, fitted_data - 1.96*sigma ];



end