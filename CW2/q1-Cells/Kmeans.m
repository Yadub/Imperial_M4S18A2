function [ Z, Mu , EK] = Kmeans( X, K, tol, display )
% Uses the K Means algorithm to segment the inputted data into K clusters
% INPUTS:      X: M x N points of data to be segmented using a gaussians
%              K: Number of clusts. Default set to 3.
% OUTPUTS:     Z: Contains the centroid vector for each data point
%              C: Defines the cluster means for each centroid vector
%             Ek: Normalized Within Cluster Sum Squared of Errors

if (nargin < 2) K = 3; end      % Set default cluster value to 3 if not given
if (nargin < 3) tol = 1e-10; end % Set default value for the break tolerance
if (nargin < 4) display = false; end % Display iteration count/means

[M,N] = size(X);    % Extract the size of K

Z = zeros(N,1);     % Array to store the indicator variables
dist = zeros(K,1);  % Array to store the norm between a point and the cluster means
size_k = zeros(K,1);% Array to store the size of each cluster in each iteration
Mu = zeros(M,K);    % Array to store the cluster means
Mu_new = rand(M,K); % Array to store the new cluster means
iters = 0;          % iteration count

while norm(Mu - Mu_new) > tol
    if (display) tic; end

    Mu = Mu_new;            % Set Mu to previously computed Cluster Center

    dist = zeros(N,K);      % Compute L2 norm
    for k = 1:K
        dist(:,k) = sum((X(:,:) - Mu(:,k)).^2,1);
    end
    [~,Z] = min(dist,[],2); % Get minimum for each N
    
    for k = 1:K             % Set new cluster centers to mean of their elements.
        if isempty(X(:,Z == k)) == 1 % Incase a cluster has no members move it to a random point
            Mu_new(:,k) = rand(M,1);
        else
            Mu_new(:,k) = mean(X(:,Z == k),2);
        end
    end
    if display
        iters = iters + 1
        Mu_new = Mu_new
        diff = norm( Mu - Mu_new)
        toc
    end
end

EK = 0; % Compute Normalized Within Cluster Sum of Squares
for k = 1:K
    EK = EK + sum( sum( (X(:,Z == k) - Mu(:,k)).^2));
end
EK = EK / N;

end

