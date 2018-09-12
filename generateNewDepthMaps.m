function [image] = generateNewDepthMaps(originalFile,x,y)
    a = dlmread(originalFile);    
    [zgrid,xgrid,ygrid] = gridfit(a(:,1),a(:,2),a(:,3),x,y);
    image = zgrid;
