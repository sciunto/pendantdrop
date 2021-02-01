import numpy as np
from numpy.linalg import pinv
from scipy import optimize


def _check_data_dim(data, dim):
    if data.ndim != 2 or data.shape[1] != dim:
        raise ValueError('Input data must have shape (N, %d).' % dim)


class BaseModel(object):

    def __init__(self):
        self.params = None


class CircleModelLinearized(BaseModel):

    """Total least squares estimator for 2D circles.

    The functional model of the circle is::

        r**2 = (x - xc)**2 + (y - yc)**2

    This estimator minimizes the squared distances from all points to the
    circle::

        min{ sum((r - sqrt((x_i - xc)**2 + (y_i - yc)**2))**2) }

    A minimum number of 3 points is required to solve for the parameters.

    Attributes
    ----------
    params : tuple
        Circle model parameters in the following order `xc`, `yc`, `r`.

    Examples
    --------
    >>> t = np.linspace(0, 2 * np.pi, 25)
    >>> xy = CircleModel().predict_xy(t, params=(2, 3, 4))
    >>> model = CircleModel()
    >>> model.estimate(xy)
    True
    >>> tuple(np.round(model.params, 5))
    (2.0, 3.0, 4.0)
    >>> res = model.residuals(xy)
    >>> np.abs(np.round(res, 9))
    array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0.])
    """

    def estimate(self, data):
        """Estimate circle model from data using total least squares.

        Parameters
        ----------
        data : (N, 2) array
            N points with ``(x, y)`` coordinates, respectively.

        Returns
        -------
        success : bool
            True, if model estimation succeeds.
        """

        _check_data_dim(data, dim=2)

        x = data[:, 0]
        y = data[:, 1]

        # http://www.had2know.com/academics/best-fit-circle-least-squares.html
        x2y2 = (x ** 2 + y ** 2)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        m1 = np.stack([[np.sum(x ** 2), sum_xy, sum_x],
                       [sum_xy, np.sum(y ** 2), sum_y],
                       [sum_x, sum_y, float(len(x))]])
        m2 = np.stack([[np.sum(x * x2y2),
                        np.sum(y * x2y2),
                        np.sum(x2y2)]], axis=-1)
        a, b, c = pinv(m1) @ m2
        a, b, c = a[0], b[0], c[0]
        xc = a / 2
        yc = b / 2
        r = np.sqrt(4 * c + a ** 2 + b ** 2) / 2

        self.params = (xc, yc, r)

        return True

    def residuals(self, data):
        """Determine residuals of data to model.
        For each point the shortest distance to the circle is returned.

        Parameters
        ----------
        data : (N, 2) array
            N points with ``(x, y)`` coordinates, respectively.

        Returns
        -------
        residuals : (N, ) array
            Residual for each data point.
        """

        _check_data_dim(data, dim=2)

        xc, yc, r = self.params

        x = data[:, 0]
        y = data[:, 1]

        return r - np.sqrt((x - xc)**2 + (y - yc)**2)

    def predict_xy(self, t, params=None):
        """Predict x- and y-coordinates using the estimated model.
        Parameters
        ----------
        t : array
            Angles in circle in radians. Angles start to count from positive
            x-axis to positive y-axis in a right-handed system.
        params : (3, ) array, optional
            Optional custom parameter set.
        Returns
        -------
        xy : (..., 2) array
            Predicted x- and y-coordinates.
        """
        if params is None:
            params = self.params
        xc, yc, r = params

        x = xc + r * np.cos(t)
        y = yc + r * np.sin(t)

        return np.concatenate((x[..., None], y[..., None]), axis=t.ndim)


class CircleModel(BaseModel):

    """Total least squares estimator for 2D circles.
    The functional model of the circle is::

        r**2 = (x - xc)**2 + (y - yc)**2

    This estimator minimizes the squared distances from all points to the
    circle::

        min{ sum((r - sqrt((x_i - xc)**2 + (y_i - yc)**2))**2) }

    A minimum number of 3 points is required to solve for the parameters.

    Attributes
    ----------
    params : tuple
        Circle model parameters in the following order `xc`, `yc`, `r`.

    Examples
    --------
    >>> t = np.linspace(0, 2 * np.pi, 25)
    >>> xy = CircleModel().predict_xy(t, params=(2, 3, 4))
    >>> model = CircleModel()
    >>> model.estimate(xy)
    True
    >>> tuple(np.round(model.params, 5))
    (2.0, 3.0, 4.0)
    >>> res = model.residuals(xy)
    >>> np.abs(np.round(res, 9))
    array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0.])

    Notes
    -----
    Implementation based on
    https://scipy-cookbook.readthedocs.io/items/Least_Squares_Circle.html
    """

    def estimate(self, data):
        """Estimate circle model from data using total least squares.

        Parameters
        ----------
        data : (N, 2) array
            N points with ``(x, y)`` coordinates, respectively.

        Returns
        -------
        success : bool
            True, if model estimation succeeds.
        """

        _check_data_dim(data, dim=2)

        x = data[:, 0]
        y = data[:, 1]

        def _f_2b(c):
            """Calculate the algebraic distance between points and
            the mean circle centered at c=(xc, yc)."""
            xc, yc = c
            Ri = np.sqrt((x-xc)**2 + (y-yc)**2)
            return Ri - Ri.mean()

        def _Df_2b(c):
            """Jacobian of f_2b
            The axis corresponding to derivatives must be coherent
            with the col_deriv option of leastsq."""
            xc, yc = c
            df2b_dc = np.empty((len(c), x.size))

            Ri = np.sqrt((x-xc)**2 + (y-yc)**2)
            df2b_dc[0] = (xc - x)/Ri  # dR/dxc
            df2b_dc[1] = (yc - y)/Ri  # dR/dyc
            df2b_dc = df2b_dc - df2b_dc.mean(axis=1)[:, np.newaxis]

            return df2b_dc

        center_estimate = np.mean(x), np.mean(y)
        (xc_2b, yc_2b), _ = optimize.leastsq(_f_2b, center_estimate,
                                             Dfun=_Df_2b,
                                             col_deriv=True)

        Ri_2b = np.sqrt((x-xc_2b)**2 + (y-yc_2b)**2)
        R_2b = Ri_2b.mean()

        self.params = (xc_2b, yc_2b, R_2b)
        self.Ri_2b = Ri_2b
        return True

    def residuals(self, data):
        """Determine residuals of data to model.

        For each point the shortest distance to the circle is returned.

        Parameters
        ----------
        data : (N, 2) array
            N points with ``(x, y)`` coordinates, respectively.

        Returns
        -------
        residuals : (N, ) array
            Residual for each data point.
        """

        _check_data_dim(data, dim=2)

        xc, yc, r = self.params

        x = data[:, 0]
        y = data[:, 1]

        return r - np.sqrt((x - xc)**2 + (y - yc)**2)

    def predict_xy(self, t, params=None):
        """Predict x- and y-coordinates using the estimated model.

        Parameters
        ----------
        t : array
            Angles in circle in radians. Angles start to count from positive
            x-axis to positive y-axis in a right-handed system.
        params : (3, ) array, optional
            Optional custom parameter set.

        Returns
        -------
        xy : (..., 2) array
            Predicted x- and y-coordinates.
        """
        if params is None:
            params = self.params
        xc, yc, r = params

        x = xc + r * np.cos(t)
        y = yc + r * np.sin(t)

        return np.concatenate((x[..., None], y[..., None]), axis=t.ndim)
