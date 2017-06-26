function [Z, Mu, LK,PKX, Sigma] = GaussianMixtureModel( X, K, tol, display, Mu, Sigma)
% Uses the EM algorithm to use the Gaussian Mixture model to fit to the
% data
% INPUTS:      X: M x N points of data to be segmented using a gaussians
%              K: Number of clusts. Default set to 3.
%             Mu: Optional inital conditions for the starting cluster means
%          Sigma: Optional inital starting cluster variance
% OUTPUTS:     Z: Contains the centroid vector for each data point
%             Mu: Defines the cluster means for each centroid vector
%             Lk: The Likelihood
%            PKX: Matrix (N X K) The probability of point X_n to be in
%            cluster k
%          Sigma: Variance matrices of the K gaussians

[M,N] = size(X); % Get dimensions of data and gaussians to be fit

if (nargin < 2) K = 3; end              % If K is not given
if (nargin < 3) tol = 1e-4; end         % If tol is not given
if (nargin < 4) display = false; end    % If display is not specified
if (nargin < 5) Mu = zeros(M,K); iMu = true; end    % If Mu not given
if (nargin < 6) Sigma = zeros(M,M,K); iSigma = true; end   % If Sigma not given

PKX = zeros(N,K); % Centroid vector for each X point
Pk = ones(K,1); % To store prior probabilities
Z = zeros(N,1);  % Inicator for each X point

% STARTING CONDITIONS: Equally spaced cluster means
if iMu == true
    for m = 1:M
        mmax = max(X(m,:));
        mmin = min(X(m,:));
        mdiff = mmax - mmin;
        Mu(m,:) = mdiff*rand(1,K);
    end
end
if iSigma == true
    for m = 1:M
        Sigma(m,m,:) = 10^(-M)*ones(1,1,K);
    end
end

% Variables to compare if clusters have stopped moving
Mu_new = Mu;
Sigma_new = Sigma;
iters = 0;
n1 = 0;
while (true)
    if (display) tic; end % Start the clock for each iteration if display is on
    
    % E-Step: Compute expectations
    for k = 1:K
        sqrt_Sigma = 1/ sqrt( det( Sigma(:,:,k) ) );
        Xminus =  X - Mu(:,k);
        invSX = -0.5 * (Sigma(:,:,k) \ Xminus);
        PKX(:,k) = Pk(k) * sqrt_Sigma * exp( dot(Xminus,invSX,1));
    end
    PKX = PKX ./ sum(PKX,2);  % Normalize posterior

    for k = 1:K
        Pk(k) = sum(PKX(:,k)) / N;
    end

    % M-Step: Maximise parameters values based on the likelihood function
    for k = 1:K
        Mu_new(:,k) = 0; % Compute new values for Cluster centers
        PKXX = X;
        for m = 1:M
            PKXX(m,:) = PKX(:,k) .* X(m,:)';
        end
        Mu_new(:,k) = Mu_new(:,k) + sum( PKXX, 2) / sum( PKX(:,k) );
        
        Sigma_new(:,:,k) = 0; % Compute new values for Cluster variances
        Xminus = X - Mu_new(:,k);
        PKXminus = Xminus;
        for m = 1:M
           PKXminus(m,:) = PKX(:,k) .* Xminus(m,:)';
        end
        Sigma_new(:,:,k) = Sigma_new(:,:,k) + PKXminus * Xminus' / sum( PKX(:,k) );
        % Incase sigma is a non invertible matrix
        if rcond(Sigma_new(:,:,k)) < 10^(-5*M)
            for m = 1:M
                Sigma_new(m,m,k) = 1;
            end
        end
    end
    
    n2 = norm(Mu_new - Mu); % norm to check if the solution has converged
    if (n2 <  tol) || (abs(n1 - n2) < tol) % Stop once converged
        Mu = Mu_new;
        Sigma = Sigma_new;
        break
    else
        n1 = n2;
        Mu = Mu_new;
        Sigma = Sigma_new;
    end

    if (display) % Display variables if asked for
        toc
        display(n2)
        iters = iters + 1;
        display(iters)
    end
end


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
    LK = LK -  (1/2) * sum( PKX(:,k)' .* dot(Xminus,invSX,1));
    LK = LK + 0.5 * Pk(k) * N * log(Pk(k));
%     LPKX = -log(PKX(:,k));
%     LPKX(LPKX == inf) = 0;
%     LK = LK + sum( PKX(:,k) .* (LPKX - (M/2) * log(2*pi) ) );
%     LK = LK + sum( PKX(:,k) .* (LPKX ) );
end

end
