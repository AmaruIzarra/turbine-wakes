import py_wake
#here we import all wake deficit models available in PyWake
import numpy as np
import matplotlib.pyplot as plt
import os
import py_wake
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from py_wake.deficit_models.deficit_model import WakeDeficitModel, BlockageDeficitModel
from py_wake.deficit_models.no_wake import NoWakeDeficit
from py_wake.site._site import UniformSite
from py_wake.flow_map import XYGrid
from py_wake.turbulence_models import CrespoHernandez
from py_wake.utils.plotting import setup_plot
from py_wake.wind_farm_models import All2AllIterative
from py_wake.deficit_models import NOJDeficit, SelfSimilarityDeficit, GCLDeficit
from py_wake.superposition_models import LinearSum
from py_wake.examples.data.hornsrev1 import V80, Hornsrev1Site
from py_wake.utils.model_utils import get_models
from py_wake.deficit_models.deficit_model import WakeDeficitModel
from py_wake.deficit_models.deficit_model import WakeDeficitModel
from numpy import newaxis as na
from shapely.geometry import Point, Polygon
from shapely.affinity import rotate, translate
from numpy import newaxis as na
from py_wake import np
from py_wake.superposition_models import SquaredSum, LinearSum
from py_wake.wind_farm_models.engineering_models import PropagateDownwind
from py_wake.turbulence_models.stf import STF2017TurbulenceModel
from py_wake.deficit_models.gaussian import NiayifarGaussianDeficit
from py_wake.deficit_models.deficit_model import DeficitModel
from py_wake.utils.gradients import cabs
from py_wake.rotor_avg_models.area_overlap_model import AreaOverlapAvgModel
from warnings import catch_warnings, filterwarnings
from py_wake.deficit_models.deficit_model import WakeRadiusTopHat
#from py_wake.deficit_models.utils import ct2a_madsen
from py_wake.utils.model_utils import DeprecatedModel


# turbine diameter
d_rot = 80


def get_flow_map(model=None, grid=XYGrid(x=np.linspace(-200, 500, 200), y=np.linspace(-200, 200, 200), h=70),
                 turbulenceModel=CrespoHernandez()):
    blockage_deficitModel = [None, model][isinstance(model, BlockageDeficitModel)]
    wake_deficitModel = [NoWakeDeficit(), model][isinstance(model, WakeDeficitModel)]
    wfm = All2AllIterative(UniformSite(), V80(), wake_deficitModel=wake_deficitModel, blockage_deficitModel=blockage_deficitModel,
                           turbulenceModel=turbulenceModel)
    return wfm(x=[0], y=[0], wd=270, ws=10, yaw=0).flow_map(grid)


def plot_deficit_map(model, cmap='Blues', levels=np.linspace(0, 10, 55)):
    fm = get_flow_map(model)
    import pandas as pd
    np.savetxt("WS_Data.csv",
               (fm.ws - fm.WS_eff).isel(h=0).data[0,:,:,0],
               delimiter =", ",  # Set the delimiter as a comma followed by a space
               fmt ='% s')  # Set the format of the data as string
    fm.plot(fm.ws - fm.WS_eff, clabel='Deficit [m/s]', levels=levels, cmap=None, normalize_with=d_rot)
    setup_plot(grid=False, ylabel="Crosswind distance [y/D]", xlabel= "Downwind distance [x/D]",
               xlim=[fm.x.min()/d_rot, fm.x.max()/d_rot], ylim=[fm.y.min()/d_rot, fm.y.max()/d_rot])


def plot_wake_deficit_map(model):
    cmap = np.r_[[[1,1,1,1],[1,1,1,1]],cm.Blues(np.linspace(-0,1,128))] # ensure zero deficit is white
    plot_deficit_map(model,cmap=ListedColormap(cmap))


class MyDeficitModel(NiayifarGaussianDeficit, WakeRadiusTopHat):

    def wake_radius(self, dw_ijlk, **_):
        return (.2*dw_ijlk)


    def create_turbine_polygon(self, xt, r_rot, wake_angle, wake_length, wind_theta):
        # master polygon: coordinate center is the turbine location
        # wake axis is the x axis
        # top of turbine rotor
        p0 = np.array([0, r_rot])
        # bottom of turbine rotor
        p3 = np.array([0, -r_rot])

        # unit vectors in the direction of wake boundaries
        rtop = np.array([np.cos(wake_angle), np.sin(wake_angle)])
        rbot = np.array([np.cos(wake_angle), -np.sin(wake_angle)])
        # wake length in the direction of the wake cone
        projected_wake_length = wake_length / np.cos(wake_angle)

        #
        p1 = p0 + rtop * projected_wake_length
        p2 = p3 + rbot * projected_wake_length

        # create, rotate, translate polygon based on wind direction
        t = Polygon([p0, p1, p2, p3])
        t = rotate(t, angle=wind_theta, origin=(0.0, 0.0), use_radians=True)
        t = translate(t, xt[0], xt[1])

        return t


MyDeficitModel.args4deficit = {'d_rot': d_rot, 'farm_size': 2000, 'z0': 0.3,
                               'wind_theta': 0}
farm_size = 10
z0 = 0.3
wind_theta = 0
zi = 60

plot_wake_deficit_map(GCLDeficit())
