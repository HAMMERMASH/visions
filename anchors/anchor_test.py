import sys
sys.path.append('../')
import numpy as np
from py_anchor import generate_base_anchor
from py_anchor import generate_original_base_anchors
from py_anchor import anchors

anchor = anchors([500,500],[36,36],16,2**np.arange(3,6),np.array([0.5,1,2]),constant_stride=16) 
print anchor.shape
print anchor
