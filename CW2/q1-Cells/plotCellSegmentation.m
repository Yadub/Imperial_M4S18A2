function [] = plotCellSegmentation( X, Y, Z, K, Mu, colour )
%MAKEIMAGE Converts the indicator varaibles and K Cluster centers, Mu, into 
% an image of size Nx by Ny.

if nargin < 6
    colour = false;
end

hold on;

if colour
    for k = 1:K
        plot(X(Z==k), Y(Z==k),'.');
    end
else
    plot(X,Y,'b.');
end

plot(Mu(1,:),Mu(2,:),'rx','LineWidth',2);

end

