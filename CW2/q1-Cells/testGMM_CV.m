% Load the image
img = imread('FluorescentCells.jpg'); 
% imshow(img);
img = double(img);
[Nx,Ny,M] = size(img);
N = Nx*Ny;
X = reshape(img, N,M)';
X = X/255;

Xperm = X(:,randperm(length(X),length(X)));
Xtest = Xperm(:,1:N/2);
Xtrain = Xperm(:, N/2+1:N);

K_vals = 2:10; 
Nk = length(K_vals);
Z = cell(Nk,1);
Ztest = cell(Nk,1);
Mu = cell(Nk,1);
PKX = cell(Nk,1);
Sigma = cell(Nk,1);
LK = zeros(Nk,1);
LKtest = zeros(Nk,1);
TimeTaken = zeros(Nk,1);

for K = K_vals
    tic;
    [Z{K-1},Mu{K-1},LK(K-1),PKX{K-1}, Sigma{K-1}] = GaussianMixtureModel(Xtrain,K);
    TimeTaken(K-1) = toc
    pkx = PKX{K-1};
    Pk = zeros(K,1);
    for k = 1:K
        Pk(k) = sum(pkx(:,k)) / length(pkx(:,k));
    end
    [ Ztest{K-1} , LKtest(K-1)] = GMMTest( Xtest, K, Mu{K-1}, Sigma{K-1}, Pk);
    
end

figure(); hold on;
plot(K_vals,-LK,'x-','DisplayName','Training fit'); 
plot(K_vals,-LKtest,'x-','DisplayName','Testing fit'); 
title('GMM: Negative LogLikelihood vs K'); xlabel('K'); ylabel('Negative LogLikelihood')
legend('show')