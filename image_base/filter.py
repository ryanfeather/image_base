import numpy as np

def is_local_max(image):
   a = image > np.roll(image,1,1)
   b = image > np.roll(image,-1, 1)
   c = image > np.roll(image,1,0)
   d = image > np.roll(image,-1,0)
   e = image > np.roll(np.roll(image,-1,1),-1,0)
   f = image > np.roll(np.roll(image,1,1),1,0)
   g = image > np.roll(np.roll(image,-1,1),1,0)
   h = image > np.roll(np.roll(image,1,1),-1,0)
   return a & b & c & d & e & f & g & h
