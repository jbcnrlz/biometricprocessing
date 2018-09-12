import pcl, numpy as np
from helper.functions import loadOBJ

if __name__ == '__main__':
	a, b, c, y = loadOBJ('/home/jbcnrlz/Documents/eurecom/EURECOM_Kinect_Face_Dataset/0001/s1/3DObj/depth_0001_s1_Neutral.obj')
	facet = pcl.PointCloud()
	facet.from_array(np.array(c,dtype=np.float32))
	print(facet.calc_normals(50))