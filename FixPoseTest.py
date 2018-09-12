import unittest
from FixPose import *
import numpy as np
import math
import logging,sys

class FixPoseTest(unittest.TestCase):

    def test_generateIndexes(self):
        fp = FixPose(None)
        expectedMiddle = [2.5,2.0]
        result = fp.getMiddlePoint(np.array([-3,5]),np.array([8,-1]))

        self.assertEqual(expectedMiddle,result)

if __name__ == '__main__':
    logging.basicConfig( stream=sys.stderr )
    logging.getLogger( "ThreeDLBP.generateCodePR" ).setLevel( logging.DEBUG )

    unittest.main()