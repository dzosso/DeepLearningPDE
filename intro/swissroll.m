function [X,Y] = swissroll(N,sigma)
    t = 2*randn(1,N)+7.5;
    Y = sign(randn(1,N));  
    X = [t.*cos(t+pi/2*Y); t.*sin(t+pi/2*Y) ] + sigma*randn(2,N);
end
