
% Load Data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load('../bitcoinData.mat');
bid = string(bid);
bid = (bid == 'TRUE');
symbol = string(symbol);
exchange = string(exchange);

X = date1;
y = price;

% Extract length of data
nx = length(X);
% Confidence interval size: 2SD = 95% for normal distribution
alpha = 0.05;

% Fit data
[ y_OLS, bounds, beta ] = OLS(X, y, alpha);

% Plot data
figure();
plotRegression(X,y,X,y_OLS,bounds,'bx');
xlabel('Date'); ylabel('Price');
title('Simple Linear Regression with 2 standard deviation bounds');

% To look at the error and if its gaussian
r = sort(y-y_OLS);
figure();
histogram(r,15,'Normalization','pdf');
xlabel('Error'); ylabel('Density'); title('Standard Linear Regression: Histogram of the Error');

% Plot corresponding pdf of the error
sigma_r = sqrt(var(r));
mu_r = mean(r);
g = exp(-(r-mu_r).^2./(2*sigma_r^2))./(sigma_r*sqrt(2*pi));
hold on;
plot(r,g,'LineWidth',1.5);

% Autocorrelation Test (Durbin-Watson test)
[p,d] = dwtest(r,X);
