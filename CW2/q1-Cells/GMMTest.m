function [ Z , LK, PKX] = GMMTest( X, K, Mu, Sigma, Pk )
% Segments the inputted test data into K clusters with means Mu
% INPUTS:      X: M x N points of data to be segmented using a gaussians
%              K: Number of clusts. Default set to 3.
%              Mu: Cluster means
%             Sigma: Cluster varainces
%               Pk: Prior probabilities of each cluster
% OUTPUTS:     Z: Contains the centroid vector for each data point
%              LK: Negative Log Likelihood
%               Probability vectors

[M,N] = size(X); % Get dimensions of data and gaussians to be fit

PKX = zeros(N,K); % Centroid vector for each X point
Z = zeros(N,1);  % Inicator for each X point

for k = 1:K % Compute Posterior
    sqrt_Sigma = 1/ sqrt( det(Sigma(:,:,k) ) );
    Xminus =  X - Mu(:,k);
    invSX = -0.5 * (Sigma(:,:,k) \ Xminus);
    PKX(:,k) = Pk(k) * sqrt_Sigma * exp( dot(Xminus,invSX,1));
end
PKX = PKX ./ sum(PKX,2); % Normalize

% Get indicator variables
for n = 1:N
    [~,index] = max(PKX(n,:));
    Z(n) = index;
end

% Compute LogLikelihood
LK = 0;
for k = 1:K
    LK = LK - Pk(k) * N * log( det( Sigma(:,:,k) ) ) / 2;
    Xminus =  X(:,:) - Mu(:,k);
    invSX = (Sigma(:,:,k) \ Xminus);
    LK = LK -  (1/2) * sum( PKX(:,k)' .*  dot(Xminus,invSX,1));
    LK = LK + 0.5 * Pk(k) * N * log(Pk(k));
%     LPKX = -log(PKX(:,k));
%     LPKX(LPKX == inf) = 0;
%     LK = LK + sum( PKX(:,k) .* (LPKX - (M/2) * log(2*pi) ) );
end

end