% Load Data 
load('bitcoinData.mat');
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
[ y_OLS, bounds, ~ ] = OLS(X, y, alpha);

% Plot data
figure(); hold on;
% plotRegression(X,y,X,y_OLS,bounds,'bx');
xlabel('Date'); ylabel('Price');
title('Simple Linear Regression with 2 standard deviation bounds');

U = unique(date1);      % Unique Days
NU = length(U);         % # Unique Days
NpDay = zeros(NU,1);    % N per Day
for day = 1:NU
    NpDay(day) = length(date1(date1 == U(day) ));
end

Bdate = date1(bid == 1);    % Dates of Bid transactions
Bprice = price(bid == 1);   % Price of Bid transactions
Bamount = amount(bid == 1); % Amount of Ask transactions
Adate = date1(bid == 0);    % Dates of Ask transactions
Aprice = price(bid == 0);   % Price of Ask transactions
Aamount = amount(bid == 0); % Amount of Ask transactions

% fig_next = figure(); hold on;
% subplot(1,2,1); plot(Bprice);
% subplot(1,2,2); plot(Aprice);

VpDay = zeros(NU,2);
AvgpDay = zeros(NU,2);

for day = 1:NU
    dayAmount = Bamount(Bdate == U(day));
    dayPrice = Bprice(Bdate == U(day));
    VpDay(day,1) = sum(dayAmount);
    AvgpDay(day,1) = sum(dayAmount.*dayPrice) / VpDay(day,1);
    
    dayAmount = Aamount(Adate == U(day));
    dayPrice = Aprice(Adate == U(day));
    VpDay(day,2) = sum(dayAmount);
    AvgpDay(day,2) = sum(dayAmount.*dayPrice) / VpDay(day,2);
end

AvgpDayT = sum(VpDay.*AvgpDay,2)./sum(VpDay')';

[ y_OLS_dayAVG, bounds, ~ ] = OLS(U, AvgpDayT, alpha);

% Plot data
plotRegression( U, AvgpDayT, U, y_OLS_dayAVG, bounds, 'bx');
plot (X,y,'k.');

