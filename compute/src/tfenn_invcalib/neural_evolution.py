"""Objective neural matrix evolution with positive resistance and sliding work."""

from dataclasses import dataclass, field

import numpy as np

from .coupled_friction_damage import FrictionDamageLaw


@dataclass(frozen=True)
class MatrixEvolutionNetwork:
    """Two-input, two-output tanh network for log resistance and flow-gap factors.

    Inputs are log(d/d_peak) and p_eff/p_ref - 1. Subtracting the hidden
    activation at (d_peak, p_ref) anchors both corrections there. All hidden
    and output weights are trainable; no measured response or specimen label
    enters the closure. The density is an irreversible history variable.
    """

    raw_parameters: np.ndarray
    density_scale: float
    pressure_scale: float = 10.

    def __post_init__(self):
        raw = np.asarray(self.raw_parameters, dtype=float)
        if raw.ndim != 1 or not raw.size or raw.size % 5 or not np.isfinite(raw).all():
            raise ValueError("Matrix network needs finite parameters for two inputs and two outputs.")
        if self.density_scale <= 0 or self.pressure_scale <= 0:
            raise ValueError("Positive density and pressure scales are required.")
        object.__setattr__(self, "raw_parameters", raw.copy())

    @classmethod
    def initialize(cls, density_scale, pressure_scale=10.):
        # Zero output weights reproduce the existing matrix law at startup.
        inputs = np.array([[.8, -.7], [.8, .7], [.3, 1.], [-.3, 1.]])
        bias = np.array([-1., -1., .5, -.5])
        return cls(np.r_[inputs.ravel(), bias, np.zeros(8)], density_scale, pressure_scale)

    def value_derivatives(self, density, effective_mean):
        """Return the two log factors and their density/pressure partials."""
        if effective_mean is None:
            raise ValueError("Neural matrix evolution requires the current effective mean stress.")
        width = len(self.raw_parameters)//5
        inputs = self.raw_parameters[:2*width].reshape(width, 2)
        bias = self.raw_parameters[2*width:3*width]
        outputs = self.raw_parameters[3*width:].reshape(2, width)
        x = np.array([np.log(density/self.density_scale), effective_mean/self.pressure_scale-1.])
        activation = np.tanh(inputs @ x+bias)
        value = outputs @ (activation-np.tanh(bias))
        jacobian = (outputs*(1-activation**2)) @ inputs
        return value, jacobian[:, 0]/density, jacobian[:, 1]/self.pressure_scale


@dataclass(frozen=True)
class NeuralFrictionDamageLaw(FrictionDamageLaw):
    """Learn matrix resistance and dilatancy while retaining their admissibility.

    R = R_base exp(g_R) > 0;
    beta = mu - (mu - beta_base) exp(g_beta) <= mu.
    On the compressive sliding surface, A:dc = lambda [c+(mu-beta)p_A]
    is nonnegative. The pressure input is tr(sigma-u I)/3, distinct from p_A,
    the mean matrix force after its energetic backstress is subtracted.
    """

    network: MatrixEvolutionNetwork = field(kw_only=True)

    def evolution(self, d, effective_mean):
        resistance, resistance_d = self.resistance(d)
        beta, beta_d = self.dilatancy(d)
        mu, mu_d = self.friction(d)
        g, gd, gp = self.network.value_derivatives(d, effective_mean)
        factors = np.exp(g)
        value = resistance*factors[0]
        gap = (mu-beta)*factors[1]
        return (value, factors[0]*(resistance_d+resistance*gd[0]), value*gp[0],
                mu-gap, mu_d-factors[1]*(mu_d-beta_d+(mu-beta)*gd[1]), -gap*gp[1])
