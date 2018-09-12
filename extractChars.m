function [K,H] = extractChars(originalFile)
    a = dlmread(originalFile);
    [Z,X,Y] = gridfit(a(:,1),a(:,2),a(:,3),10,10);
    [K,H] = surfature(X,Y,Z);
end