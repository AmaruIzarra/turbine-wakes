from py_wake.deficit_models.deficit_model import WakeDeficitModel
from numpy import newaxis as na

class MyDeficitModel(WakeDeficitModel):

    def calc_deficit(self, wind_speeds, distances, cw_ijlk,**_):
        """Calculate wake deficit caused by the x'th most upstream wind turbines
    for all wind directions(l) and wind speeds(k) on a set of points(j)

    This method must be overridden by subclass

    Arguments required by this method must be added to the class list
    args4deficit

    See class documentation for examples and available arguments

    Returns
    -------
    deficit_ijlk : array_like
    """

        # 30% deficit in downstream triangle
        ws_10pct_ijlk = 0.3*WS_ilk[:,na]
        triangle_ijlk = (self.wake_radius(dw_ijlk=dw_ijlk)>cw_ijlk)
        return ws_10pct_ijlk *triangle_ijlk

    def wake_radius(self, dw_ijlk, **_):
        return (.2*dw_ijlk)


