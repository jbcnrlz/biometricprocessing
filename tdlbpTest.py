import unittest
from tdlbp import *
from Eurecom import *
import numpy as np

import logging,sys

class Test3DLBP(unittest.TestCase):

    def test_generateIndexes(self):
        gallery = EurecomKinect('/home/jbcnrlz/Documents/eurecom/EURECOM_Kinect_Face_Dataset','s1','Depth')
        tdlbp = ThreeDLBP(3,14,[gallery])

        idxs = [(0,1),(1,2),(2,1),(1,0)]

        for i in range(5,1):
            indexes = tdlbp.generateIndexes(1,4,i)
            self.assertEqual(idxs[i],indexes)

    def test_generateCodePR(self):
        gallery = EurecomKinect('/home/jbcnrlz/Documents/eurecom/EURECOM_Kinect_Face_Dataset','s1','Depth')
        tdlbp = ThreeDLBP(3,14,[gallery])

        region = np.array([[254,253,252],[250,255,251],[254,250,249]])
        result = np.array([0,11,4,3])
        finalResult = tdlbp.generateCodePR(region,np.array([1,1]),4,1)
        self.assertEqual(result[0],finalResult[0])
        self.assertEqual(result[1],finalResult[1])
        self.assertEqual(result[2],finalResult[2])
        self.assertEqual(result[3],finalResult[3])

if __name__ == '__main__':
    logging.basicConfig( stream=sys.stderr )
    logging.getLogger( "ThreeDLBP.generateCodePR" ).setLevel( logging.DEBUG )

    unittest.main()