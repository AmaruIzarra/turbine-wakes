import csv

import pandas as pd
import py_wake
import Input_Data
from py_wake.examples.data.hornsrev1 import V80
import matplotlib.pyplot as plt
from py_wake.site import XRSite
import numpy as np
from py_wake import BastankhahGaussian
from py_wake import XYGrid, YZGrid, XZGrid
import os



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

flow_map = sim_results.flow_map()

JUMP = 5
# loop to create different z-slices
count = 0

directory = "wind_speeds"

os.makedirs(directory, exist_ok=True)

while (count <= (data.hub_height + 10)):

    grid = XYGrid(x=np.linspace(0, 700), y=np.linspace(0,700), h=count)
    flow_map = sim_results.flow_map(grid=grid, wd=data.wd, ws=data.ws)
    sum_dims_a = [d for d in ['wd', 'time', 'ws'] if d in flow_map.P.dims]
    WS_eff_a = (flow_map.WS_eff * flow_map.P / flow_map.P.sum(sum_dims_a)).sum(sum_dims_a)
    WS_eff_a.to_dataframe(name="wind_speed").reset_index().to_csv(directory + "/wind_speed_height" + str(count) +".csv")
    count += JUMP

file_list = os.listdir(directory)

df = pd.DataFrame()

for file in file_list:
    data = pd.read_csv(directory + '/' + file)
    df = pd.concat([df, data], axis=0)

df.to_csv(directory + "/total_wind_speed.csv", index=False)



# grid = XYGrid(x=np.linspace(0, 700), y=np.linspace(0,700), h=5)
# flow_map = sim_results.flow_map(grid=grid, wd=data.wd, ws=data.ws)
# flow_map.plot_wake_map()
# plt.xlabel('x [m]')
# plt.ylabel('y [m]')
# plt.title('XY Wake map for' + f' {data.ws} m/s and {data.wd} deg')
# plt.show()
#
# sum_dims_a = [d for d in ['wd', 'time', 'ws'] if d in flow_map.P.dims]
# WS_eff_a = (flow_map.WS_eff * flow_map.P / flow_map.P.sum(sum_dims_a)).sum(sum_dims_a)
#
# #print(grid)
# WS_eff_a.to_dataframe(name="wind_speed").reset_index().to_csv("file.csv")
#
# # x is the value of x that is sliced
# grid = XYGrid(x=np.linspace(0, 700), y=np.linspace(0,700), h=20)
# flow_map = sim_results.flow_map(grid=grid, wd=data.wd, ws=data.ws)
# flow_map.plot_wake_map()
# plt.xlabel('y [m]')
# plt.ylabel('z [m]')
# plt.title('YZ Wake map for' + f' {data.ws} m/s and {270} deg')
# plt.show()
#
# # y is y value to sliced
# grid = XYGrid(x=np.linspace(0, 700), y=np.linspace(0,700),h=60)
# flow_map = sim_results.flow_map(grid=grid, wd=data.wd, ws=data.ws)
# flow_map.plot_wake_map()
# plt.xlabel('x [m]')
# plt.ylabel('z [m]')
# plt.title('XZ Wake map for' + f' {data.ws} m/s and {270} deg')
# plt.show()
















