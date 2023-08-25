import pandas as pd
import py_wake
from matplotlib import pyplot as plt
from py_wake.site import XRSite
import numpy as np
from py_wake import BastankhahGaussian
from py_wake import XYGrid
import os
import my_deficit_model
import py_wake
from py_wake.examples.data.hornsrev1 import V80, Hornsrev1Site

site = Hornsrev1Site()


class Simulation():

    def __init__(self, data, windTurbines):

        if data.w_check is False:

            my_site = py_wake.site.UniformSite([1], ti=data.ti, ws=data.ws)
        else:
            my_site = py_wake.site.UniformWeibullSite(p_wd=data.p_wd, a=data.a,
                                          k=data.k, ti=data.a)
        self.my_site = my_site

        # self.wf_model = BastankhahGaussian(my_site, windTurbines)
        self.wf_model = my_deficit_model.MyDeficitModel(my_site, windTurbines)

        self.wt_x, self.wt_y = data.initial_positions

        args4deficit = ['WS_ilk', 'dw_ijlk', 'cw_ijlk']

        self.sim_results = self.wf_model()
        # self.sim_results = self.wf_model(self.wt_x, self.wt_y,
        # h=data.hub_height, ws=data.ws)

    def run(self, data):

        os.makedirs(data.directory, exist_ok=True)

        count = 0
        while count <= data.z_axis:
            grid = XYGrid(x=np.linspace(0, data.x_axis), y=np.linspace(0, data.y_axis), h=count)
            flow_map = self.sim_results.flow_map(grid=grid, wd=data.wd, ws=data.ws)
            if count == 60:
                plt.figure()
                flow_map.plot_wake_map(cmap='jet')
                plt.xlabel('x [m]')
                plt.ylabel('y [m]')
                plt.title('Wake map for'+ f' {data.wd} deg and {data.ws} m/s')
                plt.show()
            sum_dims_a = [d for d in ['wd', 'time', 'ws'] if d in flow_map.P.dims]
            WS_eff_a = (flow_map.WS_eff * flow_map.P / flow_map.P.sum(sum_dims_a)).sum(sum_dims_a)
            WS_eff_a.to_dataframe(name="wind_speed").reset_index().to_csv(data.directory + "/wind_speed_height" + str(count) +".csv")
            count += data.jump

        file_list = os.listdir(data.directory)

        df = pd.DataFrame()

        for file in file_list:
            temp_df = pd.read_csv(data.directory + '/' + file)
            df = pd.concat([df, temp_df], axis=0)

        df.to_csv(data.directory + "/total_wind_speed.csv", index=False)

