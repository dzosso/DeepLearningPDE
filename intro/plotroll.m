function plotroll(X,Y)
    scatter(X(1,Y>0), X(2,Y>0),'bo'); hold on;
    scatter(X(1,Y<0), X(2,Y<0),'rx'); hold on;
    set(gca,'XLim', [-15,15],'YLim', [-15,15] );
    daspect([1 1 1]);
end
