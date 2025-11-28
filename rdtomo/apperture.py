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

from .config import Frequencies, Beam
from .utils import warn, normalized_rmse, linear_model_str, combine_stats, compute_stats, bin_by_angle, update_nested_dict, invert_nested_dict, format_duration
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
    
class SARModel(BaseEstimator):
    def __init__(self, moco: pd.DataFrame = None, radius: str|pd.Series = "radius (m)", 
                    flight_altitude: str|pd.Series = "flight_alt (m)", azimuth: str|pd.Series = "azimuth (deg)",):
        """
        A SARModel  ...
        """
        self._radius = radius if isinstance(radius, pd.Series) else moco[radius] if radius in moco else None
        self._flight_altitude = flight_altitude if isinstance(flight_altitude, pd.Series) else moco[flight_altitude] if flight_altitude in moco else None
        self._azimuth = azimuth if isinstance(azimuth, pd.Series) else moco[azimuth] if azimuth in moco else None
        if self._radius is None or self._flight_altitude is None or self._azimuth is None:
            raise ValueError("Missing required data.")
        self.n_turns = (self._azimuth.iloc[-1] - self._azimuth.iloc[0]) / 360
        self.starting_azimuth = self._azimuth.iloc[0]
        # Bin data by angles
        vars = {
            'radius': self._radius.values,
            'flight_alt': self._flight_altitude.values,
        }
        self._binned_matrices, angle_key = bin_by_angle(self._azimuth.values, vars, units='degrees', rotate=True)
        self.angle_key = angle_key
        self.parameters = calculate_sar_parameters(self._binned_matrices, angle_key=angle_key)
        self.theta = self._azimuth.values.reshape(-1,1) - self.starting_azimuth
        # Fit initial linear models
        self.linear_models = {
            'radius': LinearRegression().fit(self.theta, self._radius.values),
            'flight_altitude': LinearRegression().fit(self.theta, self._flight_altitude.values),
        }
        self.sym_models = None # Can be generated with generate_models() but is not needed for basic usage
        # Generate predictions for all SAR parameters and fit linear models to them
        predictions, breakpoints = _predict_sar_parameters(self.linear_models, phi=self._binned_matrices[angle_key], n_turns=round(self.n_turns))
        phi = self._binned_matrices[angle_key].reshape(-1, 1)
        for band, params in self.parameters.items():
            self.linear_models[band] = {}
            for param in params.columns:
                if param == angle_key:
                    continue
                params['PRED_' + param] = predictions[band][param]
                match = re.search(r'RoI \[(.*)\] \(m\)', param)
                if match:
                    pol = match[1]
                    breakpoint = breakpoints[band][pol]
                    if breakpoint.size != 0:
                        breakpoint = breakpoint[0]
                        self.linear_models[band][param] = (
                            LinearRegression().fit(phi[:breakpoint], predictions[band][param][:breakpoint]),
                            LinearRegression().fit(phi[breakpoint:], predictions[band][param][breakpoint:]),
                            float(phi[breakpoint])
                        )
                        continue
                self.linear_models[band][param] = LinearRegression().fit(phi, predictions[band][param])

    @property
    def duration(self) -> str:
        return format_duration(self.t[-1] - self.t[0], print_days=False) 

    def fit(self, X: np.ndarray, y=None):
        warn("SARModel is pre-fitted during initialization with moco data, fit() does nothing.")
        return self # No fitting needed
    
    def evaluate(self) -> tuple[Figure, defaultdict[dict]]:
        model_evaluation = defaultdict(dict)
        fig, axs = plt.subplots(3, 6, figsize=(18, 9), squeeze=False)
        first_idx = None
        roi_idx = None
        i = 0
        for band, df in self.parameters.items():
            j = 0
            x = df[self.angle_key].values
            for col in df.columns:
                if 'PRED_' in col:
                    idx = (i,j)
                    roi = False
                    style = '-'
                    var = col[5:]
                    pred = df[col]
                    if var in df:
                        if 'RoI' in var:
                            name = 'RoI (m)'
                            if band == 'L-band':
                                roi = True
                                if roi_idx is None:
                                    roi_idx = idx
                                else:
                                    idx = roi_idx
                                if var == 'RoI [V-pol] (m)':
                                    pol = 'V-pol'
                                    style = '--'
                                if var == 'RoI [H-pol] (m)':
                                    pol = 'H-pol'
                        else:
                            name = var
                        ax = axs[*idx]
                        true = df[var]
                        rmse = root_mean_squared_error(true, pred)
                        nrmse = normalized_rmse(true,pred)
                        ax.plot(x, true, color='C0', linestyle=style, label='True' if first_idx is None and not roi else pol if roi else None)
                        ax.plot(x, pred, color='C1', linestyle=style, label='Predicted' if first_idx is None and not roi else pol if roi else None)
                        if roi:
                            ax.legend()
                        ax.set_xlabel('theta (deg)')
                        ax.set_ylabel(name)
                        ax.set_title(band)
                        model_evaluation[band][var] = {
                            'Model': linear_model_str(self.linear_models[band][var], var='theta', rounded=True) if isinstance(self.linear_models[band][var], LinearRegression) else linear_model_str(self.linear_models[band][var][0], var='theta', rounded=True) + f" for theta < {self.linear_models[band][var][2]:.2f} and else " + linear_model_str(self.linear_models[band][var][1]),
                            'RMSE': f"{rmse:.3g}",
                            'NRMSE': f"{nrmse:.3g}"
                        }
                    else:
                        raise RuntimeError(f"{band} values of {var} not found.")
                    if first_idx is None and not roi:
                        first_idx = (i,j)
                    if not roi or pol == 'H-pol':
                        j += 1
            i += 1

        handles, labels = axs[*first_idx].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncols=len(labels), bbox_to_anchor=(0.5, 1))

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        evaluation = invert_nested_dict(model_evaluation)

        return fig, evaluation
                    
    def nominalize(self) -> dict:
        return combine_stats(*compute_stats(self.parameters)) # Placeholder for converting SAR parameters to nominal values
    
    def errors(self) -> dict[str, dict[str, tuple[float, float]]]:
        """
        Returns the errors associated with the linear SAR parameter models as a nested dict with the first key specifying the band
        and the second key specifying the parameter. Each value consists of a tuple with RMSE in the first position and NRMSE in the second.
        """
        errors = self.errors
        for band, df in self.parameters.items():
            for col in df.columns:
                if "PRED_" in col:
                    val = col[5:]
                    pred = df[col]
                    if val in df.columns:
                        true = df[val]
                        rmse = root_mean_squared_error(true, pred)
                        nrmse = normalized_rmse(true,pred)
                        errors[band][val] = (rmse, nrmse)

        return errors
 
    # Generate symbolic models for all SAR parameters based on fitted linear models of radius, flight altitude and angular sampling frequency
    def generate_models(self, angle_name: str = "phi") -> dict[str, SARParaModel]:
        """
        Generate a dict of SARParaModel instances for each SAR parameter based on the fitted linear models:
        - 'VRes (m)': vertical resolution
        - 'HRes (m)': horizontal resolution
        - 'HoA (m)': height of ambiguity
        - 'RoI (m)': radius of constant illumination
        - 'BWC': relative bandwidth coverage
        - 'BWG': relative bandwidth gap number
        """
        self.sym_models = model_sar_parameters(self.linear_models, round(self.n_turns), angle_name=angle_name)
        return self.sym_models
        
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
    
    def copy(self) -> 'SARModel':
        new_model = SARModel(radius=self._radius.copy(), flight_altitude=self._flight_altitude.copy(), azimuth=self._azimuth.copy())
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
        z = self._flight_altitude

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


    def __repr__(self) -> str:
        return f"SARModel(n_turns={self.n_turns:.2f}, duration={self.duration}) fitted from (radius={self._radius}, flight_altitude={self._flight_altitude}, azimuth={self._azimuth})"
    
    def __str__(self) -> str:
        return f"SARModel over a duration of {self.duration} and {self.n_turns:.2f} turns."

def calculate_sar_parameters(binned_matrices: dict[str,np.ndarray], angle_key: str = None) -> dict[pd.DataFrame]:
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
    
    shapes = _shape_parameters(binned_matrices=binned_matrices, angle_key=angle_key)
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
        sar_parameters[band].insert(0, angle_key, shapes[angle_key])
    
    # Compute bandwidth coverage
    bwc = _compute_bandwidth_coverage(binned_matrices=binned_matrices, n_turns=shapes['n_turns'])
    # Merge the bandwidth coverage results into sar_parameters
    update_nested_dict(sar_parameters, bwc)

    return sar_parameters

def _shape_parameters(binned_matrices, angle_key) -> dict[str, np.ndarray]:
    result = {}
    result[angle_key] = binned_matrices[angle_key]
    result['n_turns'] = np.sum(~np.isnan(binned_matrices['radius']), axis=1)
    for key, matrix in binned_matrices.items():
        if key == angle_key:
            continue
        result[key+'_top'] = np.apply_along_axis(lambda row: row[~np.isnan(row)][0] if np.any(~np.isnan(row)) else np.nan, axis=1, arr=matrix)
        result[key+'_bot'] = np.apply_along_axis(lambda row: row[~np.isnan(row)][-1] if np.any(~np.isnan(row)) else np.nan, axis=1, arr=matrix)

    return result

def _compute_bandwidth_coverage(binned_matrices: dict, n_turns: np.ndarray = None) -> dict[pd.DataFrame]:
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
        n_turns = np.sum(~np.isnan(binned_matrices['radius']), axis=1)

    # Look angle
    psi = np.arctan(binned_matrices['radius'] / binned_matrices['flight_alt'])
    results = {}
    for band, bw, cf in FREQUENCIES.zip():
        f_z = cf * np.cos(psi)
        b_z = bw * np.cos(psi)
        results[band] = summarize_intervals(get_bandwidth_coverage(f_z, b_z))

    return results

def _predict_sar_parameters(models: dict, phi: np.ndarray, n_turns: int) -> dict:
    ## Constants
    # radius = k + a * phi
    k = models['radius'].intercept_
    a = models['radius'].coef_[0]
    # flight_altitude = m - b * phi
    m = models['flight_altitude'].intercept_
    b = -models['flight_altitude'].coef_[0]
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
        for i in reversed(range(n_turns)):
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


    return predictions, breakpoints

def model_sar_parameters(models: dict, n_turns: int, angle_name: str = "phi") -> dict[SARModel]:
    """
    Input: models (dict) and n_turns (the nominal number of complete turns of the spiral).
        - models contains keys "radius" and "flight_altitude"
        - each value is a LinearRegression() object fitted against drone moco.

    Output: sar_models (dict) with SARModel as values and keys:
        - 'VRes (m)': vertical resolution
        - 'HRes (m)': horizontal resolution
        - 'HoA (m)': height of ambiguity
        - 'RoI (m)': radius of constant illumination
        - 'BWC': relative bandwidth coverage
        - 'BWG': relative bandiwdth gap number
    """
    ## Constants
    # radius = k + a * phi
    k = models['radius'].intercept_
    a = models['radius'].coef_[0]
    # flight_altitude = m - b * phi
    m = models['flight_altitude'].intercept_
    b = -models['flight_altitude'].coef_[0]
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
    