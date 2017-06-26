% Load the image
img = imread('FluorescentCells.jpg'); 
% imshow(img);
img = double(img);
[Nx,Ny,M] = size(img);
N = Nx * Ny;
X = reshape(img, N ,M)';
X = X/255;
K = 3;

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

NS = length(S);
NS10 = uint32(NS/10);
Xperm = S(:,randperm(length(S),length(S)));
% Xperm = X;
Stest = Xperm(:,1:NS10);
Strain = Xperm(:, NS10+1:NS);

K_vals = 10:5:100; 
Nk = length(K_vals);
Z = cell(Nk,1);
Mu = cell(Nk,1);
Ztest = cell(Nk,1);
EK = zeros(Nk,1);
EKtest = zeros(Nk,1);
TimeTaken = zeros(Nk,1);

for i = 1:Nk
    K = K_vals(i);
    tic;
    [Z{i},Mu{i},EK(i)] = Kmeans(Strain,K);
    TimeTaken(i) = toc
    [ Ztest{i} , EKtest(i)] = KmeansTest( Stest, K, Mu{i});
end

figure(); hold on;
plot(K_vals,EK,'x-','DisplayName','Training fit'); 
plot(K_vals,EKtest,'x-','DisplayName','Testing fit'); 
title('Kmeans: Sum Squared of Errors vs K'); xlabel('K'); ylabel('Sum Squared of Errors')
legend('show')

figure();
plot(K_vals, EKtest./EK,'x-'); xlabel('K'); ylabel('Ratio between Sum Squared of Errors for training and testing');
