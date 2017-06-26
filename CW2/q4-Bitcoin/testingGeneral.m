% Load and format data
load('bitcoinData.mat');
bid = string(bid);
bid = (bid == 'TRUE');
symbol = string(symbol);
exchange = string(exchange);

N = length(price);  % get length of data

% Plot the prices the transactions at bid and ask price occurred 
fig_prices = figure(); hold on;
plot(date1(bid == 1),price(bid == 1),'rx', 'DisplayName', 'Bid Prices');
plot(date1(bid == 0),price(bid == 0),'g.', 'DisplayName', 'Ask Prices');
xlabel('Date Value'); ylabel('Price'); legend('show');
title('Bid and Ask Prices for Bitcoin Transactions')

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

% fig_next = figure(); hold on;
plot(U,AvgpDay(:,1),'x-r');
plot(U,AvgpDay(:,2),'x-g');

figure(); hold on;
plot(id(bid == 1),price(bid == 1),'rx');
plot(id(bid == 0),price(bid == 0),'gx');
xlabel('Price'); ylabel('ID'); title('Price (coloured in bid/ask) vs ID');