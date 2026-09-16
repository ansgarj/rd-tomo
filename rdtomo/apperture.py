import math
import numpy as np
import pandas as pd
import sympy as sp
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from scipy.constants import c
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import re
from pathlib import Path

from .config import Frequencies, Beam
from .utils import warn, normalized_rmse, linear_model_str, combine_stats, compute_stats, bin_by_angle, update_nested_dict, invert_nested_dict, format_duration, Angles
from .trackfinding import Spiral
from .position import Pos
FREQUENCIES = Frequencies()
BEAM = Beam()

class SARParaModel(BaseEstimator):
    def __init__(self, expr: sp.Expr):
        self.expr = expr
        self.variables = sorted(expr.free_symbols, key=lambda s: s.name)
        self._compiled = sp.lambdify(self.variables, self.expr, modules=["numpy"])

    def compile(self) -> None:
        self._compiled = sp.lambdify(self.variables, self.expr, modules=["numpy"]) # Restore after pickling

    def decompile(self) -> None:
        self._compiled = None # Useful to make SARParaModel pickle safe

    def fit(self, X: np.ndarray, y=None):
        warn("SARParaModel is pre-defined by the expression, fit() does nothing.")
        return self # No fitting needed
    
    def predict(self, X: np.ndarray, cache: bool = False):
        X = np.asarray(X).reshape(-1,1)
        result = self._compiled(*X.T)
        if cache:
            self._lastX = X
            self._predicted = result # Store original model with any negative values
        result[result < 0] = 0 # SAR parameters are non-negative
        return result

    def rmse(self, y_true: np.ndarray, y_pred: np.ndarray = None) -> np.floating:
        if y_pred is None:
            if self._predicted is None:
                raise ValueError("No cached prediction. Pass y_pred or call predict(cache=True)")
            y_pred = self._predicted
            y_pred[y_pred < 0] = 0 # SAR parameters are non-negative
        return root_mean_squared_error(y_true, y_pred)
    
    def normalized_rmse(self, y_true: np.ndarray, y_pred: np.ndarray = None) -> np.floating:
        if y_pred is None:
            if self._predicted is None:
                raise ValueError("No cached prediction. Pass y_pred or call predict(cache=True)")
            y_pred = self._predicted
            y_pred[y_pred < 0] = 0 # SAR parameters are non-negative
        return normalized_rmse(y_true, y_pred)
    
    def error(self, x: np.ndarray, y: np.ndarray) -> tuple[np.floating, np.floating]:
        self.predict(x, cache=True)
        return self.rmse(y), self.normalized_rmse(y)
    
    def linearize(self, x: np.ndarray = None):
        if x is not None:
            self.predict(x, cache=True)
        if self._lastX is None or self._predicted is None:
            raise ValueError("Either call .linearize() with an explicit x array or run .predict(cache=True) first.")
        model = LinearRegression()
        model.fit(self._lastX, self._predicted)
        linear_predict = model.predict(self._lastX)
        linear_predict[linear_predict < 0] = 0  # SAR parameters are non-negative
        linear_rmse = self.rmse(linear_predict)
        linear_nrmse = self.normalized_rmse(linear_predict)
        return model, linear_predict, (linear_rmse, linear_nrmse)

    def subs(self, **kwargs) -> sp.Expr:
        if kwargs:
            subs_dict = {}
            for sym in self.expr.free_symbols:
                if sym.name in kwargs:
                    subs_dict[sym] = kwargs[sym.name]

            expr = self.expr.subs(subs_dict)
            return expr
        return self.expr
    
    def __str__(self) -> str:
        return str(self.expr)
    
class SpiralModel(BaseEstimator):
    def __init__(self, flight: Spiral):
        """
        A SARModel  ...
        """

        self.flight = flight
        az = self.azimuth.unwrap(degrees=True)
        self.n_turns = (az[-1] - az[0]) / 360
        self.starting_azimuth = az[0]
        self.theta = az - self.starting_azimuth
        self._basic_models = {
            'radius': LinearRegression().fit(self.theta.reshape(-1,1), self.flight.radius),
            'altitude': LinearRegression().fit(self.theta.reshape(-1,1), self.flight.altitude),
        }
        self._linear_models = None
        self._parameters = None
        self._predictions = None
        self._breakpoints = None
        self._errors = None
        self._sym_models = None

        # Bin data by angles
        vars = {
            'radius': self.radius,
            'flight_alt': self.flight.altitude,
        }
        self._binned_matrices, self.angle_key = bin_by_angle(az, vars, units='degrees', rotate=True)

    @property
    def radius(self) -> np.ndarray:
        return self.flight.radius

    @property 
    def azimuth(self) -> Angles:
        return self.flight.azimuth

    @property
    def altitude(self) -> np.ndarray:
        return self.flight.altitude

    @property
    def duration(self) -> np.timedelta64:
        return self.flight.dur()

    @property
    def phi(self) -> np.ndarray:
        return self._binned_matrices[self.angle_key]

    @property
    def center(self) -> Pos:
        return self.flight.center

    @property
    def track(self) -> Pos:
        return self.flight.pos
    
    @property
    def parameters(self) -> dict[str, pd.DataFrame]:
        if not hasattr(self, '_parameters'):
            self._parameters = None
        if self._parameters is None:
            self._parameters = self._calculate_sar_parameters()
        return self._parameters

    @property
    def predictions(self) -> dict[str, pd.DataFrame]:
        if not hasattr(self, '_predictions'):
            self._predictions = None
        if self._predictions is None:
            self._predictions, self._breakpoints = self._predict_sar_parameters()
        return self._predictions

    @property
    def breakpoints(self) -> defaultdict:
        if not hasattr(self, '_breakpoints'):
            self._breakpoints = None
        if self._breakpoints is None:
            self._predictions, self._breakpoints = self._predict_sar_parameters()
        return self._breakpoints

    @property
    def basic_models(self) -> dict[str, LinearRegression]:
        if not hasattr(self, '_basic_models'):
            self._basic_models = {
                'radius': LinearRegression().fit(self.theta.reshape(-1,1), self.flight.radius),
                'altitude': LinearRegression().fit(self.theta.reshape(-1,1), self.flight.altitude),
            }
        return self._basic_models

    @property
    def linear_models(self) -> dict[str, dict[str, LinearRegression]]:
        if not hasattr(self, '_linear_models'):
            self._linear_models = None
        if self._linear_models is None:
            self._linear_models = {}
            for band, params in self.parameters.items():
                self._linear_models[band] = {}
                for param in params.columns:
                    if param == self.angle_key:
                        continue
                    match = re.search(r'RoI \[(.*)\] \(m\)', param)
                    if match:
                        pol = match[1]
                        breakpoint = self.breakpoints[band][pol]
                        if breakpoint.size != 0:
                            breakpoint = breakpoint[0]
                            self._linear_models[band][param] = (
                                LinearRegression().fit(self.phi[:breakpoint].reshape(-1,1), self.predictions[band][param][:breakpoint]),
                                LinearRegression().fit(self.phi[breakpoint:].reshape(-1,1), self.predictions[band][param][breakpoint:]),
                                float(self.phi[breakpoint])
                            )
                            continue
                    self._linear_models[band][param] = LinearRegression().fit(self.phi.reshape(-1,1), self.predictions[band][param])

        return self._linear_models

    def _get_errors(self) -> defaultdict[str, dict[str, tuple[float, float]]]:
            """
            Returns the errors associated with the linear SAR parameter models as a nested dict with the first key specifying the band
            and the second key specifying the parameter. Each value consists of a tuple with RMSE in the first position and NRMSE in the second.
            """
            errors = defaultdict(dict)
            for band, df in self.parameters.items():
                for col, val in df.items():
                    if col == self.angle_key:
                        continue
                    true = val.to_numpy()
                    pred = self.parameters[band][col].to_numpy()
                    errors[band][col] = (root_mean_squared_error(true, pred), normalized_rmse(true, pred))
    
            return errors

    @property
    def errors(self) -> dict[str, dict[str, tuple[float, float]]]:
        if not hasattr(self, '_errors'):
            self._errors = None
        if self._errors is None:
            self._errors = self._get_errors()

        return self._errors

    @property
    def sym_models(self) -> defaultdict[str, dict[str, SARParaModel]]:
        if not hasattr(self, '_sym_models'):
            self._sym_models = None
        if self._sym_models is None:
            self._sym_models = self._model_sar_parameters()

        return self._sym_models

    @classmethod
    def load(cls, path: str|Path):
        return cls(Spiral.load(path))
        
    def fit(self, X: np.ndarray, Y: None|np.ndarray = None):
        warn("SpiralModel fitting is done with flight data: fit() does nothing")
        return self # No fitting needed
    
    def evaluate(self) -> tuple[Figure, dict[str, dict[str, dict]]]:
        model_evaluation = {}
        fig, axs = plt.subplots(3, 6, figsize=(18, 9), squeeze=False)
        first_idx = None
        roi_idx = None
        i = 0

        # Basic models
        pred_radius = self.basic_models['radius'].predict(self.theta.reshape(-1,1))
        pred_alt = self.basic_models['altitude'].predict(self.theta.reshape(-1,1))
        model_evaluation["info"] = {
            "t_start": str(self.track.dt[0]),
            "t_end": str(self.track.dt[-1]),
            "center_lat": self.center.lat[0],
            "center_lon": self.center.lon[0],
            "center_easting": self.center.easting[0],
            "center_northing": self.center.northing[0],
            "radius": {
                "model": linear_model_str(self.basic_models['radius'], var='theta', rounded=True),
                "RMSE": f"{root_mean_squared_error(self.radius, pred_radius):.3g}",
                "top": f"{pred_radius[0]:.2f}",
                "bot": f"{pred_radius[-1]:.2f}",
                "max": f"{self.radius.max():.2f}",
                "min": f"{self.radius.min():.2f}"
            },
            "altitude": {
                "model": linear_model_str(self.basic_models['altitude'], var='theta', rounded=True),
                "RMSE": f"{root_mean_squared_error(self.altitude, pred_alt):.3g}",
                "top": f"{pred_alt[0]:.2f}",
                "bot": f"{pred_alt[-1]:.2f}",
                "max": f"{self.altitude.max():.2f}",
                "min": f"{self.altitude.min():.2f}"
            }
        }
        # Linear models
        for band, df in self.parameters.items():
            model_evaluation[band] = {}
            x = df[self.angle_key].to_numpy()
            j = 0
            for col, val in df.items():
                if col == self.angle_key:
                    continue
                idx = (i,j)
                style = '-'
                roi = False
                if 'RoI' in col:
                    name = 'RoI (m)'
                    if band == 'L-band':
                        roi = True
                        if roi_idx is None:
                            roi_idx = idx
                        else:
                            idx = roi_idx
                        if col == 'RoI [V-pol] (m)':
                            pol = 'V-pol'
                            style = '--'
                        if col == 'RoI [H-pol] (m)':
                            pol = 'H-pol'
                else:
                    name = col
                ax = axs[*idx]
                true = val.to_numpy()
                pred = self.predictions[band][col].to_numpy()
                rmse, nrmse = self.errors[band][col]

                ax.plot(x, true, color='C0', linestyle=style, label='True' if first_idx is None and not roi else pol if roi else None)
                ax.plot(x, pred, color='C1', linestyle=style, label='Predicted' if first_idx is None and not roi else pol if roi else None)
                if roi:
                    ax.legend()
                ax.set_xlabel('phi (deg)')
                ax.set_ylabel(name)
                ax.set_title(band)
                model_evaluation[band][col] = {
                    'Model': linear_model_str(
                            self.linear_models[band][col],
                            var='phi',
                            rounded=True
                        ) if isinstance(self.linear_models[band][col], LinearRegression) else 
                        linear_model_str(self.linear_models[band][col][0], var='phi', rounded=True) + f" for theta < {self.linear_models[band][col][2]:.2f} and else " + linear_model_str(self.linear_models[band][col][1], var='phi', rounded=True),
                    'RMSE': f"{rmse:.3g}",
                    'NRMSE': f"{nrmse:.3g}"
                }
                if first_idx is None and not roi:
                    first_idx = (i,j)
                if not roi or pol == 'H-pol':
                    j += 1
            i += 1

        handles, labels = axs[*first_idx].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncols=len(labels), bbox_to_anchor=(0.5, 1))

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # model_evaluation = invert_nested_dict(model_evaluation)

        return fig, model_evaluation
                    
    def nominalize(self) -> dict:
        return combine_stats(*compute_stats(self.parameters)) # Placeholder for converting SAR parameters to nominal values
        
    # Symbolic prediction of specified parameter, band and polarization (if applicable)
    def predict(self, X: np.ndarray, band: str, parameter: str, pol: str = "", cache: bool = False):
        if self.sym_models is None:
            raise RuntimeError("Symbolic models not generated yet. Call .generate_models() first.")
        if parameter in ('VRes', 'vres'):
            parameter = 'VRes (m)'
        if parameter in ('HRes', 'hres'):
            parameter = 'HRes (m)'
        if parameter in ('HoA', 'hoa'):
            parameter = 'HoA (m)'
        if parameter in ('RoI [H-pol]', 'roi [h-pol]'):
            parameter = 'RoI [H-pol] (m)'
        if parameter in ('RoI [V-pol]', 'roi [v-pol]'):
            parameter = 'RoI [V-pol] (m)'
        if parameter in ('RoI', 'roi', 'RoI (m)', 'roi (m)') and pol:
            if pol in ('H','h'):
                pol = 'H-pol'
            if pol in ('V', 'v'):
                pol = 'V-pol'
            parameter = f'RoI [{pol}] (m)'
        if parameter == 'bwc':
            parameter = 'BWC'
        if parameter == 'bwg':
            parameter = 'bwg'
        return self.sym_models[band][parameter].predict(X, cache=cache)
    
    # Linearization of a symbolic model of specified parameter, band and polarization (if applicable)
    def linearize(self, band: str, parameter: str, x: np.ndarray, pol: str = ""):
        if self.sym_models is None:
            raise RuntimeError("Symbolic models not generated yet. Call .generate_models() first.")
        if parameter in ('VRes', 'vres'):
            parameter = 'VRes (m)'
        if parameter in ('HRes', 'hres'):
            parameter = 'HRes (m)'
        if parameter in ('HoA', 'hoa'):
            parameter = 'HoA (m)'
        if parameter in ('RoI [H-pol]', 'roi [h-pol]'):
            parameter = 'RoI [H-pol] (m)'
        if parameter in ('RoI [V-pol]', 'roi [v-pol]'):
            parameter = 'RoI [V-pol] (m)'
        if parameter in ('RoI', 'roi', 'RoI (m)', 'roi (m)') and pol:
            if pol in ('H','h'):
                pol = 'H-pol'
            if pol in ('V', 'v'):
                pol = 'V-pol'
            parameter = f'RoI [{pol}] (m)'
        if parameter == 'bwc':
            parameter = 'BWC'
        if parameter == 'bwg':
            parameter = 'bwg'
        if x is not None:
            self.predict(x, parameter=parameter, band=band, pol=pol, cache=True)
        sym_model = self.sym_models[band][parameter]
        linear_model = self.linear_models[band][parameter]
        if sym_model._lastX is None or sym_model._predicted is None:
            raise ValueError("Either call linearize() with an explicit x array or run predict(cache=True) first.")
        new_model = LinearRegression().fit(sym_model._lastX, sym_model._predicted) 
        linear_predict = new_model.predict(sym_model._lastX)
        linear_predict[linear_predict < 0] = 0  # SAR parameters are non-negative
        linear_rmse = sym_model.rmse(linear_predict)
        linear_nrmse = sym_model.normalized_rmse(linear_predict)
        if not np.allclose(new_model.intercept_, linear_model.intercept_) or not np.allclose(new_model.coef_, linear_model.coef_):
            warn(f"Linearized model for {parameter} in {band} band differs from the directly fitted linear model: intercept({new_model.intercept_}, {linear_model.intercept_}, slope({new_model.coef_[0]}, {linear_model.coef_[0]})")
        return new_model, linear_predict, (linear_rmse, linear_nrmse)
    
    def validate(self):
        for models in self.sym_models.values():
            for model in models.values():
                model.compile()
        az = self._azimuth.values - self.starting_azimuth
        for band, models in self.linear_models.items():
            if isinstance(models, LinearRegression):
                continue
            for key in models.keys():
                self.linearize(band=band, parameter=key, x=az)
                pass
    
    def copy(self) -> 'SpiralModel':
        new_model = SpiralModel(self.flight.copy())
        new_model._parameters = self._parameters
        new_model._predictions = self._predictions
        new_model._breakpoints = self._breakpoints
        new_model._sym_models = self._sym_models
        
        return new_model
    
    def offset(self, x_offset: np.ndarray = None, y_offset: np.ndarray = None, z_offset: np.ndarray = None):
        """
        Compute relative radius r', angle theta', and altitude z' between a UAV flying a spiral path and arbitrary-shaped offset points.

        Note that the offsets can have arbitrary shapes but all provided arrays must have matching shapes.

        Returns:
        - r_prime: shape (..., len(theta))
        - theta_prime: shape (..., len(theta))
        - z_prime: shape (..., len(theta))
        """
        # UAV spiral path
        r = self._radius
        theta = self._azimuth
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = self.altitude

        # Ensure offsets are numpy arrays
        arrays = []
        if isinstance(x_offset, np.ndarray):
            arrays.append(x_offset)
        if isinstance(y_offset, np.ndarray):
            arrays.append(y_offset)
        if isinstance(z_offset, np.ndarray):
            arrays.append(z_offset)
        if len(arrays) == 0:
            raise ValueError("At least one offset array must be provided.")
        # Ensure shapes match
        shape = arrays[0].shape
        if not all(arr.shape == shape for arr in arrays):
            raise ValueError("All offset arrays must have matching shapes.")

        # Reshape UAV path for broadcasting
        x_uav = x.reshape((1,) * len(shape) + (-1,))
        y_uav = y.reshape((1,) * len(shape) + (-1,))
        z_uav = z.reshape((1,) * len(shape) + (-1,))

        # Add new axis
        arrays = [arr[..., np.newaxis] for arr in arrays]

        # Compute relative coordinates
        dx = x_uav - x_offset
        dy = y_uav - y_offset
        dz = z_uav - z_offset

        # Compute relative quantities
        r_prime = np.sqrt(dx**2 + dy**2)
        theta_prime = np.arctan2(dy, dx)
        z_prime = dz

        return r_prime, theta_prime, z_prime

    def _predict_sar_parameters(self) -> tuple[dict[str, pd.DataFrame], defaultdict]:

        models = self.basic_models
        phi = self.phi
        n_turns = self.n_turns

        # Constants
        # radius = k + a * phi
        k = models['radius'].intercept_
        a = models['radius'].coef_[0]
        # flight_altitude = m - b * phi
        m = models['altitude'].intercept_
        b = -models['altitude'].coef_[0]
        k0 = math.sqrt(math.log(2)/math.pi) # Constant for taking -3 dB resolution vertically
        n = n_turns - 1

        # t = 1 / (2aV) * (r * sqrt(a**2 + r**2) + a**2 * log(r + sqrt(a**2 + r**2)))

        # Generate predictions
        predictions = defaultdict(dict)
        breakpoints = defaultdict(dict)
        r0 = k +  a * (phi + 180 * n)
        alt0 = m - b * (phi + 180 * n)
        beta = math.atan(b / a)
        psi0 = np.atan(r0/alt0) # Mean look angle (nominal)
        l0 = 360 * n * np.sqrt(a**2 + b**2) # Maximal tomographic apperture
        p0 = np.sqrt(r0**2 + alt0**2) # Slant range at line-of-sight (nominal)
        l = l0 * np.abs(np.cos(beta - psi0)) # Effective tomographic apperture
        for band, bandwidth, central_frequency in FREQUENCIES.zip():
            bz = bandwidth*np.cos(psi0) + central_frequency*np.sin(psi0) * l/p0 # Extended vertical bandwidth
            predictions[band]['VRes (m)'] = k0*c / bz # Vertical resolution
            predictions[band]['HRes (m)'] = 1.12*c /(2*np.pi * central_frequency * np.sin(psi0)) # Horizontal resolution
            predictions[band]['HoA (m)'] = 0.5*c * n_turns * np.sin(psi0) * p0 / (l*central_frequency) # Height of ambiguity
            # Model bandwidth coverage
            previous_upper_bound = None
            for i in reversed(range(round(n_turns))):
                r = k + a * (phi + 360 * i)
                h = m - b * (phi + 360 * i)
                psi = np.atan(r / h)
                lower_bound = (central_frequency - 0.5*bandwidth) * np.cos(psi)
                upper_bound = (central_frequency + 0.5*bandwidth) * np.cos(psi)
                if previous_upper_bound is None:
                    previous_upper_bound = upper_bound
                    bwc = bandwidth * np.cos(psi)
                    gaps = np.zeros_like(phi)
                    continue
                bwc += bandwidth * np.cos(psi) - np.maximum(0, previous_upper_bound - lower_bound)
                gaps += lower_bound > previous_upper_bound
                previous_upper_bound = upper_bound    
            predictions[band]['BWC'] = bwc / bz
            predictions[band]['BWG'] = gaps / (n_turns - 1)

        for (band, pol), beamwidth, da in BEAM.zip():
            theta_far = np.pi * ((da - beamwidth/2))/180 # Far-range depression angle in radians
            theta_near = np.pi * ((90 - da - beamwidth/2))/180 # Near-range depression angle in radians 
            r1 = (m - b * (phi + 360*n))/np.tan(theta_far) - (k + a * (phi + 360*n))
            r2 = (k + a * phi) - (m - b * phi)*np.tan(theta_near)
            horizonx = theta_far < 0
            predictions[band][f'RoI [{pol}] (m)'] = r2 if horizonx else np.minimum(r1,r2)
            # Boolean array: True where r1 < r2, False where r2 <= r1
            is_r1_min = r1 < r2
            # Find where the minimum switches (i.e., where is_r1_min changes value)
            switch_point = np.where(np.diff(is_r1_min.astype(int)) != 0)[0]
            breakpoints[band][pol] = switch_point

        for band, d in predictions.items():
            predictions[band] = pd.DataFrame(d)

        return predictions, breakpoints

    def _shape_parameters(self) -> dict[str, np.ndarray]:
        result = {}
        result[self.angle_key] = self.phi
        result['n_turns'] = np.sum(~np.isnan(self._binned_matrices['radius']), axis=1)
        for key, matrix in self._binned_matrices.items():
            if key == self.angle_key:
                continue
            result[key+'_top'] = np.apply_along_axis(lambda row: row[~np.isnan(row)][0] if np.any(~np.isnan(row)) else np.nan, axis=1, arr=matrix)
            result[key+'_bot'] = np.apply_along_axis(lambda row: row[~np.isnan(row)][-1] if np.any(~np.isnan(row)) else np.nan, axis=1, arr=matrix)

        return result

    def _compute_bandwidth_coverage(self, n_turns: float|None = None) -> dict[str, pd.DataFrame]:
        # Helper function to get bandwidth coverage
        def get_bandwidth_coverage(f_array, B_array) -> list[np.ndarray]:
                """
                f_array: np.ndarray of shape (N, W) — central frequencies
                B_array: np.ndarray of shape (N, W) — bandwidths
                Returns: list of N arrays, each containing merged intervals for that row
                """
                N, W = f_array.shape
                merged_intervals_per_row = []

                for i in range(N):
                    f_row = f_array[i]
                    B_row = B_array[i]

                    # Compute lower and upper bounds
                    lower_bounds = f_row - B_row / 2
                    upper_bounds = f_row + B_row / 2

                    # Stack into intervals and sort by lower bound
                    intervals = np.stack((lower_bounds, upper_bounds), axis=1)
                    intervals = intervals[np.argsort(intervals[:, 0])]
                    intervals = intervals[~np.isnan(intervals).any(axis=1)]
                    if len(intervals) == 0:
                        merged_intervals_per_row.append(np.empty((0, 2)))
                        continue

                    # Merge overlapping intervals
                    merged = []
                    current = intervals[0]
                    for next in intervals[1:]:
                        if next[0] <= current[1] + np.finfo(float).eps:  # Overlapping or adjacent
                            current[1] = max(current[1], next[1])
                        else:
                            merged.append(current)
                            current = next
                    merged.append(current)

                    merged_intervals_per_row.append(np.array(merged))

                return merged_intervals_per_row
        
        # Helper function to summarize bandwidth
        def summarize_intervals(intervals_list) -> pd.DataFrame:
            summaries = []
            for intervals in intervals_list:
                if len(intervals) == 0:
                    summaries.append({
                        'bandwidth_coverage': 0,
                        'bandwidth_gaps': 0
                    })
                    continue

                widths = intervals[:, 1] - intervals[:, 0]
                total_width = np.nansum(widths)
                span = intervals[-1, 1] - intervals[0, 0]
                coverage = total_width / span if span > 0 else 0
                num_gaps = len(intervals) - 1

                summaries.append({
                    'BWC': coverage,
                    'BWG': num_gaps
                })
            summaries = {key: np.array([d[key] for d in summaries]) for key in summaries[0]}
            summaries['BWG'] = summaries['BWG'] / (n_turns - 1)
            return pd.DataFrame(summaries)

        if n_turns is None:
            n_turns = np.sum(~np.isnan(self._binned_matrices['radius']), axis=1)

        # Look angle
        psi = np.arctan(self._binned_matrices['radius'] / self._binned_matrices['flight_alt'])
        results = {}
        for band, bw, cf in FREQUENCIES.zip():
            f_z = cf * np.cos(psi)
            b_z = bw * np.cos(psi)
            results[band] = summarize_intervals(get_bandwidth_coverage(f_z, b_z))

        return results

    def _calculate_sar_parameters(self) -> dict[str, pd.DataFrame]:
        """
        Input: shapes (pd.DataFrame) with columns:
            - angle_name
            - 'radius_top'
            - 'radius_bot'
            - 'flight_alt_top'
            - 'flight_alt_bot'
            - 'n_turns'

        Output: a dict with keys 'C-band', 'L-band', 'P-band' with pd.DataFarmes containing columns:
            - angle_name
            - 'VRes (m)': vertical resolution
            - 'HRes (m)': horizontal resolution
            - 'HoA (m)': height of ambiguity
            - 'RoI [pol] (m)': radius of constant illumination, with 'pol' indicating the available polarizations
            - 'BWC': relative bandwidth coverage
            - 'BWG': relative bandwidth gap number 
        """
        
        shapes = self._shape_parameters()
        sar_parameters = defaultdict(dict)

        # Calculate help values
        dr = shapes['radius_bot'] - shapes['radius_top'] # Radius variation
        r0 = (shapes['radius_top'] + shapes['radius_bot'])/2 # Mean radius
        dalt = shapes['flight_alt_top'] - shapes['flight_alt_bot'] # Altitude variation
        alt0 = (shapes['flight_alt_top'] + shapes['flight_alt_bot'])/2 # Mean altitude
        beta = np.atan(dalt/dr) # Tomographic apperture slant
        psi0 = np.atan(r0/alt0) # Mean look angle (nominal)
        l0 = np.sqrt(dr**2 + dalt**2) # Maximal tomographic apperture
        p0 = np.sqrt(r0**2 + alt0**2) # Slant range at line-of-sight (nominal)
        l = l0 * abs(np.cos(beta - psi0)) # Effective tomographic apperture
        k0 = np.sqrt(np.log(2)/np.pi) # Constant for taking -3 dB resolution vertically

        # Calculate frequency dependent parameters
        for band, b, f in FREQUENCIES.zip():
            bz = b * np.cos(psi0) + f * np.sin(psi0) * l/p0 # Extended vertical bandwidth (nominal)
            sar_parameters[band]['VRes (m)'] = k0*c / bz # Vertical -3 dB resolution (nominal)
            sar_parameters[band]['HRes (m)'] = 1.12 * c / (f * 2*np.pi * np.sin(psi0)) # Horizontal -3 dB resolution (nominal)
            sar_parameters[band]['HoA (m)'] = shapes['n_turns'] * np.sin(psi0) * c * p0 / (2 * l * f) # Height of ambiguity

        # Calculate beam shape dependent parametrs
        for (band, pol), bw, da in BEAM.zip():
            theta_far = np.deg2rad((da - bw/2)) # Far-range depression angle
            theta_near = np.deg2rad((90 - da - bw/2)) # Near-range depression angle
            r1 = np.maximum(0, shapes['flight_alt_bot'] / np.tan(theta_far) - shapes['radius_bot']) # Limit at the base of the flight path
            r2 = np.maximum(0, shapes['radius_top'] - shapes['flight_alt_top'] * np.tan(theta_near)) # Limit at the top of the flight path 
            horizonx = theta_far < 0 # The beam crosses the horizon
            sar_parameters[band][f'RoI [{pol}] (m)'] = r2 if horizonx else np.minimum(r1,r2)

        # Convert to a dict of DataFrames
        for band, d in sar_parameters.items():
            sar_parameters[band] = pd.DataFrame(d)
            sar_parameters[band].insert(0, self.angle_key, shapes[self.angle_key])
        
        # Compute bandwidth coverage
        bwc = self._compute_bandwidth_coverage(n_turns=shapes['n_turns'])

        # Merge the bandwidth coverage results into sar_parameters
        update_nested_dict(sar_parameters, bwc)

        return sar_parameters

    def _model_sar_parameters(self, angle_name: str = "phi") -> defaultdict[str, dict[str, SARParaModel]]:
        """
        Input: models (dict) and n_turns (the nominal number of complete turns of the spiral).
            - models contains keys "radius" and "altitude"
            - each value is a LinearRegression() object fitted against drone moco.

        Output: sar_models (dict) with SARParaModel as values and keys:
            - 'VRes (m)': vertical resolution
            - 'HRes (m)': horizontal resolution
            - 'HoA (m)': height of ambiguity
            - 'RoI (m)': radius of constant illumination
            - 'BWC': relative bandwidth coverage
            - 'BWG': relative bandiwdth gap number
        """
        models = self.basic_models
        n_turns = round(self.n_turns)

        ## Constants
        # radius = k + a * phi
        k = models['radius'].intercept_
        a = models['radius'].coef_[0]
        # flight_altitude = m - b * phi
        m = models['altitude'].intercept_
        b = -models['altitude'].coef_[0]
        k0 = math.sqrt(math.log(2)/math.pi) # Constant for taking -3 dB resolution vertically
        n = n_turns - 1

        # t = 1 / (2aV) * (r * sqrt(a**2 + r**2) + a**2 * log(r + sqrt(a**2 + r**2)))

        # Generate symbolic expressions
        expr = defaultdict(dict)
        phi = sp.Symbol(angle_name, real=True, nonnegative=True) # Wrapped angle
        r0 = k +  a * (phi + 180 * n)
        alt0 = m - b * (phi + 180 * n)
        beta = math.atan(b / a)
        psi0 = sp.atan(r0/alt0) # Mean look angle (nominal)
        l0 = 360 * n * sp.sqrt(a**2 + b**2) # Maximal tomographic apperture
        p0 = sp.sqrt(r0**2 + alt0**2) # Slant range at line-of-sight (nominal)
        l = l0 * sp.Abs(sp.cos(beta - psi0)) # Effective tomographic apperture
        for band, bandwidth, central_frequency in FREQUENCIES.zip():
            bz = bandwidth*sp.cos(psi0) + central_frequency*sp.sin(psi0) * l/p0 # Extended vertical bandwidth
            expr[band]['VRes (m)'] = k0*c / bz # Vertical resolution
            expr[band]['HRes (m)'] = 1.12*c /(2*sp.pi * central_frequency * sp.sin(psi0)) # Horizontal resolution
            expr[band]['HoA (m)'] = 0.5*c * n_turns * sp.sin(psi0) * p0 / (l*central_frequency) # Height of ambiguity
            # Model bandwidth coverage
            previous_upper_bound = None
            for i in reversed(range(n_turns)):
                r = k + a * (phi + 360 * i)
                h = m - b * (phi + 360 * i)
                psi = sp.atan(r / h)
                lower_bound = (central_frequency - 0.5*bandwidth) * sp.cos(psi)
                upper_bound = (central_frequency + 0.5*bandwidth) * sp.cos(psi)
                if previous_upper_bound is None:
                    previous_upper_bound = upper_bound
                    bwc = bandwidth * sp.cos(psi)
                    gaps = 0
                    continue
                bwc += bandwidth * sp.cos(psi) - sp.Max(0, previous_upper_bound - lower_bound)
                gaps += sp.Piecewise(
                    (1, lower_bound > previous_upper_bound),
                    (0, True)
                )
                previous_upper_bound = upper_bound    
            expr[band]['BWC'] = bwc / bz
            expr[band]['BWG'] = gaps / (n_turns - 1)
        for (band, pol), beamwidth, da in BEAM.zip():
            theta_far = sp.pi * ((da - beamwidth/2))/180 # Far-range depression angle in radians
            theta_near = sp.pi * ((90 - da - beamwidth/2))/180 # Near-range depression angle in radians 
            r1 = (m - b * (phi + 360*n))/sp.tan(theta_far) - (k + a * (phi + 360*n))
            r2 = (k + a * phi) - (m - b * phi)*sp.tan(theta_near)
            horizonx = theta_far < 0
            expr[band][f'RoI [{pol}] (m)'] = sp.Piecewise(
                (r2, horizonx),
                (sp.Min(r1,r2), True)
            )


        # Create sar_models dict
        sar_model = defaultdict(dict)
        for band, expressions in expr.items():
            for key, expr in expressions.items():
                sar_model[band][key] = SARParaModel(expr=expr)

        return sar_model

    def __repr__(self) -> str:
        return f"SpiralModel(n_turns={self.n_turns:.2f}, duration={self.duration}) fitted from (radius={self.radius}, altitude={self.altitude}, azimuth={self.azimuth})"
    
    def __str__(self) -> str:
        return f"SpiralModel over a duration of {self.duration} and {self.n_turns:.2f} turns."


## Model spiral tracks
# def model_spirals(tracks, path, dry, verbose, npar: int = os.cpu_count()):
#     with Pool(processes=npar) as pool:
#         results = pool.starmap(_model, [(i, track, dry) for i, track in tracks.items()])

#         for i, fig, evaluation in sorted(results, key=lambda x: x[0]):
#             if verbose:
#                 print(f"Spiral {i}:", end=" ", flush=True)
#                 print(json.dumps(evaluation, indent=4))
#             if not dry:
#                 fig_path = path.with_name(path.stem + f"-{i:02}_spiral_model.pdf")
#                 eval_path = fig_path.with_suffix(".json")
#                 fig.savefig(fig_path, format="pdf")
#                 with open(eval_path, 'w') as dst:
#                     json.dump(evaluation, dst, indent=4)
#                 print(f"Model evaluation for Spiral {i} saved to {fig_path} and {eval_path}")

# def _model(i: int, track: pd.DataFrame, dry: bool = False) -> tuple[int, Figure, defaultdict[dict]]:
#     model = SARModel(track)
#     fig, evaluation = model.evaluate()
#     try:
#         fig.canvas.manager.set_window_title(f"SAR parameters: Spiral {i}")
#     except Exception:
#         pass
#     if dry:
#         if i == 1:
#             print("Showing model plots ...", end=" ", flush=True)
#         plt.show()
#         if i == 1:
#             print("done.")

#     return i, fig, evaluation


# if variance:  # propagate variance
#         # Extract variances
#         r_min_var, r_max_var, alt_min_var, alt_max_var, N_turns_var = extract_keys(variance, 'r_min', 'r_max', 'alt_min', 'alt_max', 'N_turns')

#         # Variances of intermediate quantities
#         dr_var = r_min_var + r_max_var
#         dalt_var = alt_min_var + alt_max_var
#         r0_var = 0.25 * (r_min_var + r_max_var)
#         alt0_var = 0.25 * (alt_min_var + alt_max_var)

#         # Exact derivative for beta = atan(dalt / dr)
#         beta_var = (1 / (1 + (dalt / dr)**2))**2 * (
#             (dalt / dr**2)**2 * dr_var + (1 / dr)**2 * alt_min_var
#         )

#         # Exact derivative for psi0 = atan(r0 / alt0)
#         psi0_var = (1 / (1 + (r0 / alt0)**2))**2 * (
#             (1 / alt0**2) * r0_var + (r0**2 / alt0**4) * alt0_var
#         )

#         # Variance of l0 and p0
#         l0_var = (dr / l0)**2 * dr_var + (dalt / l0)**2 * dalt_var
#         p0_var = (r0 / p0)**2 * r0_var + (alt0 / p0)**2 * alt0_var

#         # Variance of effective aperture l
#         l_var = abs(math.cos(beta - psi0))**2 * l0_var + (l0 * math.sin(beta - psi0))**2 * (beta_var + psi0_var)

#         # Per-band variance propagation
#         for band, b, f in FREQUENCIES.zip():
#             bz = b * math.cos(psi0) + f * math.sin(psi0) * l / p0

#             # Variance of bz
#             bz_var = (
#                 (b * -math.sin(psi0) + f * math.cos(psi0) * l / p0)**2 * psi0_var +
#                 (f * math.sin(psi0) / p0)**2 * l_var +
#                 (f * math.sin(psi0) * l / p0**2)**2 * p0_var
#             )

#             # dz variance
#             sar_variance[band]['VRes'] = (k * c / bz**2)**2 * bz_var

#             # dxy variance
#             dxy_var = (1.12 * c * math.cos(psi0) / (f * 2 * math.pi * math.sin(psi0)**2))**2 * psi0_var
#             sar_variance[band]['HRes'] = dxy_var

#             # h_amb variance
#             h_amb_var = (c / (2 * f))**2 * (
#                 (math.sin(psi0) * p0 / l)**2 * N_turns_var +
#                 (N_turns * math.cos(psi0) * p0 / l)**2 * psi0_var +
#                 (N_turns * math.sin(psi0) / l)**2 * p0_var +
#                 (N_turns * math.sin(psi0) * p0 / l**2)**2 * l_var
#             )
#             sar_variance[band]['HoA'] = h_amb_var

#         # r_ci variance propagation
#         for (band, pol), bw in BEAM.zip():
#             theta_far = np.deg2rad(DEPRESSION_ANGLE - bw / 2)
#             theta_near = np.deg2rad(90 - DEPRESSION_ANGLE - bw / 2)
#             r1 = max(0, alt_min / np.tan(theta_far) - r_max)
#             r2 = max(0, r_min - alt_max * np.tan(theta_near))
#             horizonx = theta_far < 0

#             # r1 variance
#             r1_var = (1 / np.tan(theta_far))**2 * alt_min_var + r_max_var

#             # r2 variance
#             r2_var = (np.tan(theta_near))**2 * alt_max_var + r_min_var

#             # Final r_ci variance
#             sar_variance[band][f'RI [{pol}]'] = r2_var if horizonx else (r1_var if r1 < r2 else r2_var)
    