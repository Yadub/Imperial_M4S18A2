% Load the image
img = imread('FluorescentCells.jpg'); 
% imshow(img);
img = double(img);
[Nx,Ny,M] = size(img);
N = Nx * Ny;
X = reshape(img, N ,M)';
X = X/255;
K = 3;

load('Kmeans_counting.mat');
% [Z,C] = Kmeans(X,K);
Z = Z1;
C = C1;


% To print out the segment being converted
% % I = reshape(Z,Nx,Ny);
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

K_vals = 80:2:100;
Nk = length(K_vals);
Z = cell(Nk,1);
Mu = cell(Nk,1);
LK = zeros(Nk,1);
TimeTaken = zeros(Nk,1);

for i = 1:Nk
    K = K_vals(i);
    tic;
    [Z{i},Mu{i},LK(i)] = Kmeans(S,K,1e-14);
    timet = toc
    TimeTaken(i) = timet;
end

BIC = N * log(LK / N) + K_vals.* log(N);

figure();
plotCellSegmentation(Xp,Yp,Z{i},K,640*Mu{i}, false)
axis([0 Nx 0 Ny])

figure();
plot(K_vals,LK);