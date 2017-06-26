function [ ] = plotRegression( X, y, xstar, bestEstimate, bounds, linestyle,  dateFormat )
%PLOTGP Summary of this function goes here
%   Detailed explanation goes here

if nargin < 6
    linestyle = 'b+';
end
if nargin < 7
    dateFormat = 0;
end

hold on
if dateFormat ~= 0
    xstar = datetime(datestr(xstar,0));
    X = datetime(datestr(X,0));
end

stdRegion(xstar,bounds);
plot (xstar,bestEstimate,'c','DisplayName','Best Estimate');
plot (X,y,linestyle,'DisplayName','Data Points');
legend('show');
hold off;
end

