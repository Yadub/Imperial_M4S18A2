img = imread('FluorescentCells.jpg'); % Load the image and format it
img = double(img);
[Nx,Ny,M] = size(img);
X = reshape(img, Nx*Ny,M)';
X = X/255;
% Initalize arrays needed to store data
K_vals = 2:10;
Nk = length(K_vals);
Z = cell(Nk,1);
Mu = cell(Nk,1);
EK = zeros(Nk,1);
TimeTaken = zeros(Nk,1);
% Compute K Means clustering for each value of K
for K = K_vals
    tic;
    [Z{K-1},Mu{K-1},EK(K-1)] = Kmeans(X,K);
    TimeTaken(K-1) = toc
end
% Plot results
figure();
subplot(1,4,1);
plotImage(Z{2-1},Mu{2-1},2,Nx,Ny);
title(['Image Segmentation using K-Means, K: ',num2str(2)]);
subplot(1,4,2);
plotImage(Z{3-1},Mu{3-1},3,Nx,Ny);
title(['Image Segmentation using K-Means, K: ',num2str(3)]);
subplot(1,4,3);
plotImage(Z{5-1},Mu{5-1},5,Nx,Ny);
title(['Image Segmentation using K-Means, K: ',num2str(5)]);
subplot(1,4,4);
plotImage(Z{10-1},Mu{10-1},10,Nx,Ny);
title(['Image Segmentation using K-Means, K: ',num2str(10)]);

figure();
plot(K_vals,EK,'x-'); 
title('Normalized Squared Sum of Errors vs K'); xlabel('K'); ylabel('Normalized Squared Sum of Errors')
