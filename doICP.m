function [matrix] = doICP(path)
	obj = read_wobj(strcat(path,'face_normal.obj'));
	obj2 = read_wobj(strcat(path,'face_mirror.obj'));
	[newface, matrix] = ICP_finite(obj.vertices,obj2.vertices);