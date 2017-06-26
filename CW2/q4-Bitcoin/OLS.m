function [ fitted_data, bounds, beta_coeffs, sigma ] = OLS(x, y, alpha)
%OLS - Ordinary Least Squares
%   Detailed explanation goes here

if nargin < 3, alpha = 0.05; end

% Extract length of the data of interest
N = length(x);

xm = mean(x);
ym = mean(y);

beta1 = sum((x - xm).*(y-ym));
beta1 = beta1 / sum( (x - xm).^2 );
beta0 = ym - beta1 * xm;
beta_coeffs = [beta0, beta1];
% Find the estimated values
fitted_data = beta_coeffs(2) * x + beta_coeffs(1);

r = fitted_data-y;
SSe = sum(r.^2);         % Squared sum of errors
var_e = SSe/(N-2);       % Variance of the error
sigma = sqrt(var_e);
tval = tinv(1 - alpha/2, N-2); % Student t value with given degrees of freedom and (1-alpha) Confidence interval
c = sqrt( 1 + 1/N + (( x - xm ).^2) / sum( ( x - xm ).^2 )  );
bound = tval * sigma * c;

bounds = [fitted_data +  bound, fitted_data - bound ];

end