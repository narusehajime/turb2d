import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
from turb2d import RunMultiFlows
import time

# ipdb.set_trace()

proc = 10  # number of processors to be used
num_runs = 100
Cmin, Cmax = [0.001, 0.03]
rmin, rmax = [50., 200.]
hmin, hmax = [25., 150.]

C_ini = np.random.uniform(Cmin, Cmax, num_runs)
r_ini = np.random.uniform(rmin, rmax, num_runs)
h_ini = np.random.uniform(hmin, hmax, num_runs)

rmf = RunMultiFlows(
    C_ini,
    r_ini,
    h_ini,
    'test191209_01.nc',
    processors=proc,
    endtime=4000.0,
)
rmf.create_datafile()
start = time.time()
rmf.run_multiple_flows()
print("elapsed time: {} sec.".format(time.time() - start))
