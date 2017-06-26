% Load the image
img = imread('FluorescentCells.jpg'); 
% imshow(img);
img = double(img);
[Nx,Ny,M] = size(img);
N = Nx*Ny;
X = reshape(img, N,M)';
X = X/255;

Xperm = X(:,randperm(length(X),length(X)));
% Xperm = X;
Xtest = Xperm(:,1:N/2);
Xtrain = Xperm(:, N/2+1:N);

K_vals = 2:10; 
Nk = length(K_vals);
Z = cell(Nk,1);
Ztest = cell(Nk,1);
Mu = cell(Nk,1);
EK = zeros(Nk,1);
EKtest = zeros(Nk,1);
TimeTaken = zeros(Nk,1);

for K = K_vals
    tic;
    [Z{K-1},Mu{K-1},EK(K-1)] = Kmeans(Xtrain,K);
    TimeTaken(K-1) = toc
    [ Ztest{K-1} , EKtest(K-1)] = KmeansTest( Xtest, K, Mu{K-1} );
end

figure(); hold on;
plot(K_vals,EK,'x-','DisplayName','Training fit'); 
plot(K_vals,EKtest,'x-','DisplayName','Testing fit'); 
title('The normalised within cluster sum of squares vs K'); xlabel('K'); ylabel('Normalised within cluster sum of squares')
legend('show')

figure();
plot(K_vals, EK/EKtest)