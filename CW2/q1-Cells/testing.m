% Load the image
img = imread('FluorescentCells.jpg'); 

% To see the image if need be
% subplot(1,2,1);
imshow(img); title('Inital image file')

% Convert image to double for working
% img = double(img);

imgSegmentationGMM('FluorescentCells',4)

% img_LAB = rgb2lab(img);
% cform = makecform('srgb2lab');
% img_LAB = applycform(img,cform);
% normalize the color pixel
% I = img_LAB;
% I = double(I)/255;
% figure; imagesc(I); title('image plotted in Lab color space');

% imshow(img);
img = double(img);
img = img/255; % Normalize
K = 3; % Num clusters

[Nx, Ny, ~] = size(img);
Np = Nx * Ny;
[X,Y,Z] = sphere(16);
xvals = reshape(img(:,:,1),Np,1);
yvals = reshape(img(:,:,2),Np,1);
zvals = reshape(img(:,:,3),Np,1);
S = 5*ones(Np,1);
C = xvals.*yvals.*zvals;
s = S(:);
c = C(:);
size(xvals)
size(s)
scatter3(xvals,yvals,zvals,s,c);

I = zeros(Nx,Ny);
dist = zeros(K,1);
size_k = zeros(K,1);
C = zeros(3,K);
C_new = rand(3,K);

c1 = img(2,2,:); c1 = c1(:); C_new(:,1) = c1(:);
c2 = img(10,24,:); c2 = c2(:); C_new(:,2) = c2(:);
c3 = img(10,50,:); c3 = c3(:); C_new(:,3) = c3(:);

while norm(C - C_new) > 1e-1
    
    C = C_new;
    C_new = zeros(3,K);
    
    for x = 1:Nx
        for y = 1:Ny
            for k = 1:K
                colors = img(x,y,:);
                colors = colors(:);
                dist(k) = norm( colors - C(:,k) );
            end
            [~,i] = min(dist);
            I(x,y) = i;
            size_k(i) = size_k(i) + 1;
            C_new(:,i) = C_new(:,i) + colors(:);
        end
    end

    for k = 1:K
        if size_k(k) ~= 0 
            C_new(:,k) = C_new(:,k) / size_k(k);
        end
    end
end

% Convert cluster data back to image data
img_K = zeros(Nx,Ny,3);
for x = 1:Nx
    for y = 1:Ny
        for k = 1:K
            if I(x,y) == k
                img_K(x,y,1) = C(1,k);
                img_K(x,y,2) = C(2,k);
                img_K(x,y,3) = C(3,k);
            end
        end
    end
end

% Convert back to ints to plot
img_K = img_K * 255;
img_U = uint8(img_K);
img_Ud = double(img_K);
norm(img_K(:,:,1) - img_Ud(:,:,1))
subplot(1,2,2);
imshow(img_U);


