function [matrix] = doICP(path_original,path_destination)
	obj = read_wobj(path_original);
	obj2 = read_wobj(path_destination);
	[newface, matrix] = ICP_finite(obj.vertices,obj2.vertices);