import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from py_wake.examples.data.hornsrev1 import V80
from numpy import newaxis as na
from py_wake.superposition_models import SquaredSum
from py_wake.wind_farm_models.engineering_models import PropagateDownwind
from py_wake.deficit_models.deficit_model import DeficitModel
from py_wake.rotor_avg_models.area_overlap_model import AreaOverlapAvgModel
from py_wake.deficit_models.deficit_model import WakeRadiusTopHat
from py_wake.utils.model_utils import DeprecatedModel
from py_wake.site._site import UniformSite
from py_wake.flow_map import HorizontalGrid
from py_wake.wind_turbines import WindTurbines
from py_wake.deficit_models.deficit_model import WakeDeficitModel


# turbine diameter
d_rot = 80


class MyDeficitModel(WakeDeficitModel):


    def ct2a_madsen(self, ct, ct2ap=np.array([0.2460, 0.0586, 0.0883])):
        """
        BEM axial induction approximation by
        Madsen, H. A., Larsen, T. J., Pirrung, G. R., Li, A., and Zahle, F.: Implementation of the blade element momentum model on a polar grid and its aeroelastic load impact, Wind Energ. Sci., 5, 1–27, https://doi.org/10.5194/wes-5-1-2020, 2020.
        """
        return ct * (ct2ap[0] + ct * (ct2ap[1] + ct * ct2ap[2]))

    def __init__(self, ct2a=ct2a_madsen, k=.1, groundModel=None):
        DeficitModel.__init__(self, groundModel=groundModel,
                          use_effective_ws=False, use_effective_ti=False)
        self.a = [0, k]
        self.ct2a = ct2a

    def k_ilk(self, **kwargs):
        """Wake expansion factor for the i'th turbine for all
        wind directions(l) and wind speeds(k) at a set of points(j)"""

        TI_ref_ilk = kwargs[self.TI_key]
        k_ilk = self.a[0] * TI_ref_ilk + self.a[1]
        return k_ilk

    def _calc_layout_terms(self, D_src_il, wake_radius_ijl, dw_ijlk, cw_ijlk, **kwargs):

        WS_ref_ilk = kwargs[self.WS_key]
        R_src_il = D_src_il / 2

        term_denominator_ijlk = np.where(dw_ijlk > 0, ((wake_radius_ijl / R_src_il[:, na, :])**2)[..., na], 1)

        in_wake_ijlk = wake_radius_ijl[..., na] > cw_ijlk

        self.layout_factor_ijlk = WS_ref_ilk[:, na] * (in_wake_ijlk / term_denominator_ijlk)

    def calc_deficit(self, ct_ilk, **kwargs):
        if not self.deficit_initalized:
            self._calc_layout_terms(ct_ilk=ct_ilk, **kwargs)
        term_numerator_ilk = 2. * self.ct2a(ct_ilk)
        return term_numerator_ilk[:, na] * self.layout_factor_ijlk

    def wake_radius(self, D_src_il, dw_ijlk, **kwargs):
        if 'TI_eff_ilk' not in kwargs:
            kwargs['TI_eff_ilk'] = 0.0
            kwargs['TI_ilk'] = 0.0
        k_ijlk = np.atleast_3d(self.k_ilk(**kwargs))[:, na]
        wake_radius_ijlk = (k_ijlk * dw_ijlk + D_src_il[:, na, :, na] / 2)
        return wake_radius_ijlk


class MyJensen(PropagateDownwind, DeprecatedModel):

    def ct2a_madsen(self, ct, ct2ap=np.array([0.2460, 0.0586, 0.0883])):
        """
        BEM axial induction approximation by
        Madsen, H. A., Larsen, T. J., Pirrung, G. R., Li, A., and Zahle, F.: Implementation of the blade element momentum model on a polar grid and its aeroelastic load impact, Wind Energ. Sci., 5, 1–27, https://doi.org/10.5194/wes-5-1-2020, 2020.
        """
        return ct * (ct2ap[0] + ct * (ct2ap[1] + ct * ct2ap[2]))

    def __init__(self, site, windTurbines,
                 ct2a=ct2a_madsen, k=.1, superpositionModel=SquaredSum(), deflectionModel=None, turbulenceModel=None,
                 groundModel=None):
        """
        Parameters
        ----------
        site : Site
            Site object
        windTurbines : WindTurbines
            WindTurbines object representing the wake generating wind turbines
        k : float, default 0.1
            wake expansion factor
        superpositionModel : SuperpositionModel, default SquaredSum
            Model defining how deficits sum up
        blockage_deficitModel : DeficitModel, default None
            Model describing the blockage(upstream) deficit
        deflectionModel : DeflectionModel, default None
            Model describing the deflection of the wake due to yaw misalignment, sheared inflow, etc.
        turbulenceModel : TurbulenceModel, default None
            Model describing the amount of added turbulence in the wake
        """
        PropagateDownwind.__init__(self, site, windTurbines,
                                   wake_deficitModel=MyDeficitModel(
                                       k=k, ct2a=ct2a, groundModel=groundModel),
                                   superpositionModel=superpositionModel,
                                   deflectionModel=deflectionModel,
                                   turbulenceModel=turbulenceModel)
        # DeprecatedModel.__init__(self, 'py_wake.literature.noj.Jensen_1983')

    def calculate_new_wake_model(self, site, windTurbines, xt, yt, x, y, hub_height, ws):

        wake_model = MyJensen(site, windTurbines, ct2a=self.ct2a_madsen, superpositionModel=SquaredSum())
        sim_res = wake_model(x=xt, y=yt, wd=[270], ws=[ws])
        WS_eff_xy = sim_res.flow_map(HorizontalGrid(x=x, y=y, h=hub_height)).WS_eff_xylk.mean(['wd', 'ws'])
        WS_eff_xy.to_dataframe(name="wind_speed").reset_index().to_csv("C:/Users/amaru/PycharmProjects/turbine-wakes/wind_speed_height.csv")
        fig, ax = plt.subplots(1, 1)
        # plots contour lines
        cmap = mpl.cm.hot
        contour = WS_eff_xy.plot.contourf(levels=100, cmap=cmap)
        ax.contour
        ax.set_title('Contour Plot')
        ax.set_xlabel('feature_x')
        ax.set_ylabel('feature_y')

        plt.show()
        return WS_eff_xy.values


