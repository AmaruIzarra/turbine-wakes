import matplotlib.pyplot as plt
import matplotlib as mpl
from numpy import newaxis as na
from py_wake.superposition_models import SquaredSum
from py_wake.wind_farm_models.engineering_models import PropagateDownwind
from py_wake.deficit_models.deficit_model import DeficitModel
from py_wake.flow_map import HorizontalGrid
from py_wake.deficit_models.deficit_model import WakeDeficitModel
import warnings


def get_dU(x, r, R, CT, TI):
    """Computes the wake velocity deficit at a location

    Inputs
    ----------
    x: float
        Distance between turbines and wake location in the wind direction
    r: float
        Radial distance between the turbine and the location
    R: float
        Wake producing turbine's radius [m]
    CT: float
        Outputs WindTurbine object's thrust coefficient
    TI: float
        Ambient turbulence intensity [-]
    order: int, optional

    Returns
    -------
    dU: float
        Wake velocity deficit at a location
    """

    CT = np.maximum(CT, np.finfo(float).eps)
    Area = np.pi * R * R
    Rw, xx0, c1 = get_Rw(x, R, TI, CT)
    c1s = c1 * c1

    term10 = (1 / 9)
    term20 = my_power(CT * Area / (xx0 * xx0), 1. / 3.)

    term310 = my_power(r, 1.5)
    term320 = 1.0 / np.sqrt(3. * c1s * CT * Area * xx0)
    term30 = term310 * term320
    term41 = my_power(35. / (2. * np.pi), .3)
    term42 = my_power(3. * c1s, -0.2)
    term40 = term41 * term42
    t4 = term30 - term40
    dU1 = -term10 * term20 * t4 * t4

    dU = dU1

    dU = np.where((Rw < r) | (x <= 0), 0, dU)
    return dU


def get_Rw(x, R, TI, CT):
    """Computes the wake radius at a location.
    [1]-eq.3

    .. math::
        R_w = \\left(\\frac{105  c_1^2 }{2 \\pi}\\right)^{0.2} (C_T A (x + x_0))^{1/3}

    with A, the area, and x_0 and c_1 defined as

    .. math::
        x_0 = \\frac{9.6 D}{\\left(\\frac{2 R_96}{k D} \\right)^3 - 1}

        c_1 = \\left(\\frac{k D}{2}\\right)^{5/2}
              \\left(\\frac{105}{2 \\pi} \\right)^{-1/2}
              (C_T A x_0)^{-5/6}

    with k and m defined as

    .. math::
        k = \\sqrt{\\frac{m + 1}{2}}

        m = \\frac{1}{\\sqrt{1 - C_T}}

    Inputs
    ----------
    x: float or ndarray
        Distance between turbines and wake location in the wind direction
    R: float
        Wind turbine radius
    TI: float
        Ambient turbulence intensity
    CT: float
        Outputs WindTurbine object's thrust coefficient

    Returns
    -------
    Rw: float or ndarray
        Wake radius at a location
    """
    D = 2.0 * R
    Area = np.pi * R * R

    m = 1.0 / (np.sqrt(1.0 - CT))
    k = np.sqrt((m + 1.0) / 2.0)

    R96 = get_r96(D, CT, TI)
    x0 = (9.6 * D) / (my_power(2.0 * R96 / (k * D), 3.0) - 1.0)
    xx0 = x + x0
    term1 = my_power(k * D / 2.0, 2.5)
    term2 = my_power(105.0 / (2.0 * np.pi), -0.5)
    term3 = my_power(CT * Area * x0, -5.0 / 6.0)
    c1 = term1 * term2 * term3

    Rw = my_power(105.0 * c1 * c1 / (2.0 * np.pi), 0.2) * my_power(CT * Area * xx0, 1.0 / 3.0)

    Rw = np.where(x + x0 <= 0., 0., Rw)
    return Rw, xx0, c1


def my_power(term, factor):

    return np.exp(factor * np.log(term))


def get_r96(D, CT, TI):
    """Computes the wake radius at 9.6D downstream location of a turbine from empirical relation

    .. math::
        R_{9.6D} = a_1 \\exp (a_2 C_T^2 + a_3 C_T + a_4)  (b_1  TI + b_2)  D

    Inputs
    ----------
    D: float
        Wind turbine diameter
    CT: float
        Outputs WindTurbine object's thrust coefficient
    TI: float
        Ambient turbulence intensity
    pars: list
        GCL Model parameters [a1, a2, a3, a4, b1, b2]

    Returns
    -------
    R96: float
        Wake radius at 9.6D downstream location
    """
    a1, a2, a3, a4, b1, b2 = [0.435449861, 0.797853685, -0.124807893, 0.136821858, 15.6298, 1.0]
    R96 = a1 * (np.exp(a2 * CT * CT + a3 * CT + a4)) * (b1 * TI + b2) * D

    return R96


class MyDeficitModel(WakeDeficitModel):

    def __init__(self, use_effective_ws=False, use_effective_ti=False, rotorAvgModel=None, groundModel=None):
        DeficitModel.__init__(self, rotorAvgModel=rotorAvgModel, groundModel=groundModel,
                              use_effective_ws=use_effective_ws, use_effective_ti=use_effective_ti)

    def wake_radius(self, dw_ijlk, D_src_il, ct_ilk, **kwargs):

        TI_ilk = kwargs[self.TI_key]
        with warnings.catch_warnings():
            # if term is 0, exp(log(0))=0 as expected for a positive factor
            warnings.filterwarnings('ignore', r'invalid value encountered in log')
            return get_Rw(x=dw_ijlk, R=(D_src_il / 2)[:, na, :, na], TI=TI_ilk[:, na], CT=ct_ilk[:, na])[0]

    def calc_deficit(self, D_src_il, dw_ijlk, cw_ijlk, ct_ilk, **kwargs):

        WS_ilk = kwargs[self.WS_key]
        TI_ilk = kwargs[self.TI_key]
        eps = 1e-10
        dw_ijlk_gt0 = np.maximum(dw_ijlk, eps)
        R_src_il = D_src_il / 2.
        dU = -get_dU(x=dw_ijlk_gt0, r=cw_ijlk, R=R_src_il[:, na, :, na], CT=ct_ilk[:, na], TI=TI_ilk[:, na])
        return WS_ilk[:, na] * dU


class MyLarsen(PropagateDownwind):

    def __init__(self, site, windTurbines, rotorAvgModel=None, superpositionModel=SquaredSum(),
                 deflectionModel=None, turbulenceModel=None, groundModel=None):

        PropagateDownwind.__init__(self, site, windTurbines,
                                   wake_deficitModel=MyDeficitModel(rotorAvgModel=rotorAvgModel,
                                                                groundModel=groundModel),
                                   superpositionModel=superpositionModel,
                                   deflectionModel=deflectionModel, turbulenceModel=turbulenceModel)

    def calculate_new_Larsen(self, site, windTurbines, xt, yt, x, y, hub_height, ws):

        wake_model = MyLarsen(site, windTurbines, superpositionModel=SquaredSum())
        sim_res = wake_model(x=xt, y=yt, wd=[270], ws=[ws])
        WS_eff_xy = sim_res.flow_map(HorizontalGrid(x=x, y=y, h=hub_height)).WS_eff_xylk.mean(['wd', 'ws'])
        fig, ax = plt.subplots(1, 1)
        # plots contour lines
        cmap = mpl.cm.hot
        contour = WS_eff_xy.plot.contourf(levels=100, cmap=cmap)
        ax.contour
        ax.set_title('Contour Plot')
        ax.set_xlabel('feature_x')
        ax.set_ylabel('feature_y')

        plt.show()
        print(WS_eff_xy.values)
        return WS_eff_xy.values


from py_wake.site._site import UniformSite
from py_wake.examples.data.hornsrev1 import V80
import numpy as np

site = UniformSite()
wt = V80()
xt = [1000, 500] # x-coordinates of windturbines
yt = [1000, 500] # y-coordinates of windturbines
ws = 13
x = np.linspace(0, 2000, 400)
y = np.linspace(0, 2000, 400)
hub_height = 80

MyLarsen(site, wt).calculate_new_Larsen(site, wt, xt, yt, x, y, hub_height, ws)



