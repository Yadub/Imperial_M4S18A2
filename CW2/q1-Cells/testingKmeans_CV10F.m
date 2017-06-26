for i = 2:9
    Xtest = X(:,1+(i-1)*N/10:i*N/10);
    Xtrain = X(:,1+(i-1)*N/10:(i-1)*N/10 (i+1)*N/10+1:N);

    for K = K_vals
        tic;
        [Z{K-1},Mu{K-1},EK(K-1)] = Kmeans(Xtrain,K);
        TimeTaken(K-1) = toc
        [ Ztest{K-1} , EKtest(K-1)] = KmeansTest( Xtest, K, Mu{K-1} );
    end
