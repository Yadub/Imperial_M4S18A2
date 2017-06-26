% Load the image
img = imread('FluorescentCells.jpg'); 
% imshow(img);
img = double(img);
[Nx,Ny,M] = size(img);
N = Nx * Ny;
X = reshape(img, N ,M)';
X = X/255;
K = 3;

load('Kmeans_counting.mat'); % Has the data stored for the segmentation with K = 3
tic;
% [Z1,C1] = GaussianMixtureModel(X,K,1e-3); % Data can also be found using
toc
Z = Z1;
C = C1;

Z = reshape(Z, Nx, Ny);

S = [];
Xp = []; Yp = [];
for x = 1:Nx
    for y = 1:Ny
        if Z(x,y) == 2
            S = [S, [x;y]];
            Xp = [Xp, x];
            Yp = [Yp, y];
        end
    end
end
S(1,:) = S(1,:) / Nx;
S(2,:) = S(2,:) / Ny;

K_vals = 10:10:100;
Nk = length(K_vals);
Z = cell(Nk,1);
Mu = cell(Nk,1);
LK = zeros(Nk,1);
TimeTaken = zeros(Nk,1);

for i = 1:Nk
    K = K_vals(i);
    tic;
    [Z{i},Mu{i},LK(i)] = GaussianMixtureModel(S,K,1e-3);
    timet = toc
    TimeTaken(i) = timet;
end

BIC = N * log(LK / N) + K_vals.* log(N);

figure();
subplot(1,3,1);
plotCellSegmentation(Xp,Yp,Z{i},K,640*Mu{i}, false)
title('GMM: Cluster Centers. K = 100'); xlabel('x'); ylabel('y')
axis([0 Nx 0 Ny])
subplot(1,3,2);
plotCellSegmentation(Xp,Yp,Z{i-2},80,640*Mu{i-2}, false)
title('GMM: Cluster Centers. K = 80'); xlabel('x'); ylabel('y')
axis([0 Nx 0 Ny])
subplot(1,3,3);
plotCellSegmentation(Xp,Yp,Z{i-4},60,640*Mu{i-4}, false)
title('GMM: Cluster Centers. K = 100'); xlabel('x'); ylabel('y')
axis([0 Nx 0 Ny])

figure();
plot(K_vals,-LK,'x-'); 
title('GMM: Negative LogLikelihood vs K (Cell Counting)'); xlabel('K'); ylabel(' Negative LogLikelihood')
