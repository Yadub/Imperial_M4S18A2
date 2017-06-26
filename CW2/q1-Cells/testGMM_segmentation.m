img = imread('FluorescentCells.jpg'); % Load the image and format it 
img = double(img);
[Nx,Ny,M] = size(img);
N = Nx * Ny;
X = reshape(img, N,M)';
X = X/255;
% Initalize arrays needed to store data
K_vals = 2:10; 
Nk = length(K_vals);
Z = cell(Nk,1);
Mu = cell(Nk,1);
LK = zeros(Nk,1);
TimeTaken = zeros(Nk,1);

% scatter3(X(1,:),X(2,:),X(3,:),[],sum(X)); xlabel('R'),ylabel('G'),zlabel('B')
% title('Scatter Plot in the RBG space of the Image pixels')

% Compute Gaussian Mixture Model for each value of K
for K = K_vals
    tic;
    [Z{K-1}, Mu{K-1}, LK(K-1), Proba, Sigma] = GaussianMixtureModel(X,K,1e-3);
    TimeTaken(K-1) = toc
end

Mu1 = Mu{K-1};
Cvec = reshape(Proba, Nx, Ny, K);
img_K = zeros(Nx,Ny,M);
for x = 1:Nx
    for y = 1:Ny
        for k = 1:K
                img_K(x,y,1) = img_K(x,y,1) + Cvec(x,y,k) * Mu1(1,k);
                img_K(x,y,2) = img_K(x,y,2) + Cvec(x,y,k) * Mu1(2,k);
                img_K(x,y,3) = img_K(x,y,3) + Cvec(x,y,k) * Mu1(3,k);
        end
    end
end
% Plot results
img_K = uint8(img_K * 255);
figure();
subplot(1,2,1);
imshow(img_K);
title('Image reproduced using probabilites (soft clustering) K = 10');
subplot(1,2,2);
plotImage(Z{K-1},Mu{K-1},K,Nx,Ny);
title('Image reproduced using indicator variables (hard clustering) K = 10');

figure();
plot(K_vals,-LK,'x-'); 
title('GMM: Negative LogLikelihood vs K'); xlabel('K'); ylabel(' Negative LogLikelihood')

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
