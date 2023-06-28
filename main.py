import py_wake
import Input_Data
from py_wake.examples.data.hornsrev1 import V80
import matplotlib.pyplot as plt
from py_wake.site import XRSite
import numpy as np
from py_wake import BastankhahGaussian
from py_wake import XYGrid, YZGrid, XZGrid


data = Input_Data.InputData('input_data.csv')
windTurbines = V80()

if data.w_check is False:
    my_site = py_wake.site.UniformSite([1], ti=data.ti, ws=data.ws)
else:
    my_site = py_wake.site.UniformWeibullSite(p_wd=data.p_wd, a=data.a,
                                              k=data.k, ti=data.a)

wf_model = BastankhahGaussian(my_site, windTurbines)
wt_x, wt_y = data.initial_positions

sim_results = wf_model(wt_x, wt_y, h=data.hub_height, ws=data.ws)


grid = XYGrid(x=np.linspace(0, 700), y=np.linspace(0,700))
flow_map = sim_results.flow_map(grid=grid, wd=data.wd, ws=data.ws)
flow_map.plot_wake_map()
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title('XY Wake map for' + f' {data.ws} m/s and {data.wd} deg')
plt.show()

# x is the value of x that is sliced
grid = YZGrid(x=600, y=None, resolution=100)
flow_map = sim_results.flow_map(grid=grid, wd=270, ws=data.ws)
flow_map.plot_wake_map()
plt.xlabel('y [m]')
plt.ylabel('z [m]')
plt.title('YZ Wake map for' + f' {data.ws} m/s and {270} deg')
plt.show()

# y is y value to sliced
grid = XZGrid(y=100, resolution=100)
flow_map = sim_results.flow_map(grid=grid, wd=data.wd, ws=data.ws)
flow_map.plot_wake_map()
plt.xlabel('x [m]')
plt.ylabel('z [m]')
plt.title('XZ Wake map for' + f' {data.ws} m/s and {270} deg')
plt.show()
















