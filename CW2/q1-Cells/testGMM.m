% Load the image
img = imread('FluorescentCells.jpg'); 
% imshow(img);
img = double(img);
[Nx,Ny,M] = size(img);
X = reshape(img, Nx*Ny,M)';
X = X/255;
K = 3;
% [Z,Mu] = GaussianMixtureModel( X, K);
[GMM,Mu,Z] = GaussianMixtureModel1( X, K);

% I = reshape(Z,Nx,Ny);
Cvec = reshape(GMM, Nx, Ny, K);
Z =  reshape(Z, Nx, Ny);
% Convert cluster data back to image data
img_K = zeros(Nx,Ny,M);
for x = 1:Nx
    for y = 1:Ny
        for k = 1:K
                img_K(x,y,1) = img_K(x,y,1) + Cvec(x,y,k) * Mu(1,k);
                img_K(x,y,2) = img_K(x,y,2) + Cvec(x,y,k) * Mu(2,k);
                img_K(x,y,3) = img_K(x,y,3) + Cvec(x,y,k) * Mu(3,k);
        end
    end
end
% for x = 1:Nx
%     for y = 1:Ny
%         for k = 1:K
%             if Z(x,y) == k
%                 img_K(x,y,1) = Mu(1,k);
%                 img_K(x,y,2) = Mu(2,k);
%                 img_K(x,y,3) = Mu(3,k);
%             end
%         end
%     end
% end
img_K = uint8(img_K * 255);
figure();
imshow(img_K);