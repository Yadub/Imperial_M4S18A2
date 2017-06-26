function [] = makeImage( Z, Mu, K, Nx, Ny )
%MAKEIMAGE Converts the indicator varaibles and K Cluster centers, Mu, into 
% an image of size Nx by Ny.

Z = reshape(Z, Nx, Ny);

% Convert cluster data back to image data
img_K = zeros(Nx,Ny,3);
for x = 1:Nx
    for y = 1:Ny
        for k = 1:K
            if Z(x,y) == k
                img_K(x,y,1) = Mu(1,k);
                img_K(x,y,2) = Mu(2,k);
                img_K(x,y,3) = Mu(3,k);
            end
        end
    end
end
img_K = uint8(img_K * 255);
imshow(img_K);

end

