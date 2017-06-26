% Load the image
img = imread('FluorescentCells.jpg'); 
% imshow(img);
img = double(img);
[Nx,Ny,M] = size(img);
X = reshape(img, Nx*Ny,M)';
X = X/255;
K = 3;

% tic;
[z,mu,ek] = Kmeans(X,K);
% time_taken = toc

% I = reshape(Z,Nx,Ny);
% Output image
makeImage(z,mu,K,Nx,Ny);