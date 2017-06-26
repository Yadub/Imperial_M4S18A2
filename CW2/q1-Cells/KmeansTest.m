function [ Z , EK] = KmeansTest( X, K, Mu )
% Segments the inputted test data into K clusters with means Mu
% INPUTS:      X: M x N points of data to be segmented using a gaussians
%              K: Number of clusts. Default set to 3.
%              Mu: Cluster centers
% OUTPUTS:     Z: Contains the centroid vector for each data point
%              EK: Goodness of Clustering

[~,N] = size(X); % Extract the size of K

dist = zeros(N,K); % Assign to cluster with smallest L2 norm
for k = 1:K
    dist(:,k) = sum((X(:,:) - Mu(:,k)).^2,1);
end
[~,Z] = min(dist,[],2);

EK = 0; % Compute Goodness of fit
for k = 1:K
    EK = EK + sum( sum( (X(:,Z == k) - Mu(:,k)).^2));
end
EK = EK/N;

end

