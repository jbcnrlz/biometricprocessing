from EurecomTemplate import *
import numpy as np

class TemplateNode(EurecomTemplate):    

    graph = None

    def createGraph(self,pointCenterNode,size=3):
        self.graph = None
        facet = pcl.PointCloud()
        facet.from_array(np.array(self.image,dtype=np.float32))
        kdeTree = facet.make_kdtree_flann()
        indices, sqr_distances = kdeTree.nearest_k_search_for_point(facet, pointCenterNode, size)
        neighboursData = []
        currRegion = []
        for j in indices:
            currRegion.append(facet[j])
            indicesInternal, sqr_distances = kdeTree.nearest_k_search_for_point(facet, j, size+1)
            clusterData = []
            for i in indicesInternal[1:]:
                clusterData.append(facet[i])
            neighboursData.append(clusterData)

        self.graph = Node(facet[pointCenterNode],neighboursData,currRegion)

class Node:

    def __init__(self,current,neighbours,point):
        self.point = point
        self.current = current
        self.neighbours = neighbours

    def __str__(self):
        print(self.point)
        print(self.current)
        print(self.neighbours)
        return ''