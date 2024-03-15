import numpy as np 
import sys
fbin1 = sys.argv[-1]
fbin2 = sys.argv[-2]
fbin1 = np.fromfile(fbin1,dtype="double")
fbin2 = np.fromfile(fbin2,dtype="double")
diff = np.abs(fbin2-fbin1)
print(np.amax(diff))
