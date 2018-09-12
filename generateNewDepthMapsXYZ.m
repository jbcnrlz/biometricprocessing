function [zgrid,xgrid,ygrid] = generateNewDepthMapsXYZ(cloudPoint)
    a = cloudPoint;
    [zgrid,xgrid,ygrid] = gridfit(a(:,1),a(:,2),a(:,3),100,100);
