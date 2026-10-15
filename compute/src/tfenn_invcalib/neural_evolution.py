"""Objective neural matrix evolution with positive resistance and sliding work."""

from dataclasses import dataclass, field

import numpy as np
from scipy.special import expit

from .coupled_friction_damage import FrictionDamageLaw


@dataclass(frozen=True)
class MatrixEvolutionNetwork:
    """Anchored tanh network for log resistance and flow-gap factors.

    Inputs are log(d/d_peak) and p_eff/p_ref - 1. Subtracting the hidden
    activation at (d_peak, p_ref) anchors both corrections there. The optional
    third input is the objective bedding projection of the normalized
    deviatoric effective stress or matrix force supplied by the constitutive
    update. Its reference value is zero at zero deviatoric input.
    All hidden and output weights are trainable; no measured response or specimen label
    enters the closure. The density is an irreversible history variable.

    With local_width > 0, parameters contain an anchored base network followed
    by an unanchored two-input local network, two output offsets, and gate
    coordinates (logit center, log slope). The local output is multiplied by
    G(F), the tanh gate normalized to G(0)=0 and G(1)=1.

    Optional fabric_knots replace the direct angular dependence by C1 convex
    interpolation of these complete log-factor maps. Including F=0 preserves
    the hydrostatic anchor. No new fitting coordinates are introduced.
    """

    raw_parameters: np.ndarray
    density_scale: float
    pressure_scale: float = 10.
    input_size: int = 2
    local_width: int = field(default=0, kw_only=True)
    fabric_knots: tuple = field(default=(), kw_only=True)

    def __post_init__(self):
        raw = np.asarray(self.raw_parameters, dtype=float)
        if self.input_size not in (2, 3):
            raise ValueError("Matrix evolution uses two state inputs, optionally with fabric direction.")
        base_count = raw.size-(5*self.local_width+4 if self.local_width else 0)
        if self.local_width < 0 or (self.local_width and self.input_size != 3):
            raise ValueError("A separable local map requires the fabric input and a positive width.")
        if (raw.ndim != 1 or base_count <= 0 or base_count % (self.input_size+3)
                or not np.isfinite(raw).all()):
            raise ValueError("Matrix network parameters must match its inputs and two outputs.")
        if self.density_scale <= 0 or self.pressure_scale <= 0:
            raise ValueError("Positive density and pressure scales are required.")
        knots = tuple(self.fabric_knots)
        if knots and (self.input_size != 3 or len(knots) < 2
                      or not np.isfinite(knots).all() or np.any(np.diff(knots) <= 0)
                      or 0. not in knots):
            raise ValueError("Fabric knots must increase and include the reference F=0 in a three-input map.")
        object.__setattr__(self, "raw_parameters", raw.copy())
        object.__setattr__(self, "fabric_knots", knots)

    @property
    def base_width(self):
        local_count = 5*self.local_width+4 if self.local_width else 0
        return (len(self.raw_parameters)-local_count)//(self.input_size+3)

    @property
    def hidden_width(self):
        return self.base_width+self.local_width

    def parameter_indices(self, block):
        """Coordinates of the two output channels or the fabric dependence."""
        width = self.base_width
        base_count = width*(self.input_size+3)
        if block == "fabric":
            indices = list(range(2, self.input_size*width, self.input_size))
            return indices + (list(range(len(self.raw_parameters)-2, len(self.raw_parameters)))
                              if self.local_width else [])
        if block in ("flow", "response"):
            channels = (1,) if block == "flow" else (0, 1)
            indices = [i for k in channels for i in
                       range((self.input_size+1+k)*width, (self.input_size+2+k)*width)]
            if self.local_width:
                local = self.local_width
                indices += [i for k in channels for i in
                            range(base_count+(3+k)*local, base_count+(4+k)*local)]
                indices += [base_count+5*local+k for k in channels]
            return indices
        return list(range(len(self.raw_parameters)))

    @classmethod
    def initialize(cls, density_scale, pressure_scale=10., input_size=2):
        # Zero output weights reproduce the existing matrix law at startup.
        inputs = np.array([[.8, -.7], [.8, .7], [.3, 1.], [-.3, 1.]])
        bias = np.array([-1., -1., .5, -.5])
        if input_size == 3:
            inputs = np.column_stack((inputs, np.zeros(4)))
        return cls(np.r_[inputs.ravel(), bias, np.zeros(8)], density_scale, pressure_scale, input_size)

    def with_fabric_input(self):
        """Embed an existing law exactly, with zero new input weights."""
        if self.input_size == 3:
            return self
        width = len(self.raw_parameters)//5
        inputs = self.raw_parameters[:2*width].reshape(width, 2)
        raw = np.r_[np.column_stack((inputs, np.zeros(width))).ravel(),
                    self.raw_parameters[2*width:]]
        return type(self)(raw, self.density_scale, self.pressure_scale, 3)

    def with_separable_features(self, local_width, gate_center=.8125, gate_slope=24.):
        """Move appended angular features to a gated density/pressure map.

        Their F=1 response and the hydrostatic reference anchor are preserved.
        Gate parameters and local offsets remain trainable coordinates.
        """
        if self.local_width or self.input_size != 3 or not 0 < local_width < self.base_width:
            raise ValueError("Splitting features requires a dense fabric network and both base and local units.")
        if not 0 < gate_center < 1 or not np.isfinite(gate_slope) or gate_slope <= 0:
            raise ValueError("The gate center must be between zero and one and its slope positive.")
        width, base = self.base_width, self.base_width-local_width
        w = self.raw_parameters[:3*width].reshape(width, 3)
        b = self.raw_parameters[3*width:4*width]
        out = self.raw_parameters[4*width:].reshape(2, width)
        raw = np.r_[w[:base].ravel(), b[:base], out[:, :base].ravel(),
                    w[base:, :2].ravel(), b[base:]+w[base:, 2], out[:, base:].ravel(),
                    -out[:, base:] @ np.tanh(b[base:]),
                    np.log(gate_center/(1-gate_center)), np.log(gate_slope)]
        return MatrixEvolutionNetwork(raw, self.density_scale, self.pressure_scale, 3,
                                      local_width=local_width, fabric_knots=self.fabric_knots)

    def value_derivatives(self, density, effective_mean, fabric_projection=0.):
        """Log factors and density, mean-stress and fabric-projection partials."""
        if effective_mean is None:
            raise ValueError("Neural matrix evolution requires the current effective mean stress.")
        if self.fabric_knots:
            # Interpolate the constitutive functions at the supplied angular
            # supports, not observed stress/pressure curves. Smoothstep is a
            # C1 convex blend with constant extension beyond the supports.
            knots = self.fabric_knots
            j = int(np.clip(np.searchsorted(knots, fabric_projection)-1, 0, len(knots)-2))
            left, right = knots[j:j+2]
            t = float(np.clip((fabric_projection-left)/(right-left), 0., 1.))
            weight, slope = t*t*(3-2*t), 6*t*(1-t)/(right-left)
            a = self._direct_value_derivatives(density, effective_mean, left)
            b = self._direct_value_derivatives(density, effective_mean, right)
            return (*(x+weight*(y-x) for x, y in zip(a[:3], b[:3])), slope*(b[0]-a[0]))
        return self._direct_value_derivatives(density, effective_mean, fabric_projection)

    def _direct_value_derivatives(self, density, effective_mean, fabric_projection):
        width = self.base_width
        offset = self.input_size*width
        inputs = self.raw_parameters[:offset].reshape(width, self.input_size)
        bias = self.raw_parameters[offset:offset+width]
        base_count = (self.input_size+3)*width
        outputs = self.raw_parameters[offset+width:base_count].reshape(2, width)
        x = np.array([np.log(density/self.density_scale), effective_mean/self.pressure_scale-1.])
        if self.input_size == 3:
            x = np.r_[x, fabric_projection]
        activation = np.tanh(inputs @ x+bias)
        value = outputs @ (activation-np.tanh(bias))
        jacobian = (outputs*(1-activation**2)) @ inputs
        fabric_derivative = np.zeros(2)
        if self.input_size == 3:
            fabric_derivative = jacobian[:, 2]
        if self.local_width:
            # The local density/pressure transitions have no angular input.
            # G(0)=0 preserves the anchor while G(1)=1 fixes its normalization.
            local = self.local_width
            raw = self.raw_parameters[base_count:]
            weights = raw[:2*local].reshape(local, 2)
            local_bias = raw[2*local:3*local]
            local_outputs = raw[3*local:5*local].reshape(2, local)
            center, slope = expit(raw[-2]), np.exp(raw[-1])
            a0, a1 = np.tanh(-slope*center), np.tanh(slope*(1-center))
            a = np.tanh(slope*(fabric_projection-center))
            gate = (a-a0)/(a1-a0)
            gate_derivative = slope*(1-a*a)/(a1-a0)
            activation = np.tanh(weights @ x[:2]+local_bias)
            local_value = local_outputs @ activation+raw[5*local:5*local+2]
            value += gate*local_value
            jacobian[:, :2] += gate*((local_outputs*(1-activation**2)) @ weights)
            fabric_derivative = fabric_derivative+gate_derivative*local_value
        return value, jacobian[:, 0]/density, jacobian[:, 1]/self.pressure_scale, fabric_derivative


@dataclass(frozen=True)
class NeuralFrictionDamageLaw(FrictionDamageLaw):
    """Learn matrix resistance and dilatancy while retaining their admissibility.

    R = R_base exp(g_R) > 0;
    beta = mu - (mu - beta_base) exp(g_beta) <= mu.
    On the compressive sliding surface, A:dc = lambda [c+(mu-beta)p_A]
    is nonnegative. The pressure input is tr(sigma-u I)/3, distinct from p_A,
    the mean matrix force after its energetic backstress is subtracted.
    With resistance_uses_pressure=False, g_R is evaluated at p_ref while
    g_beta retains the current pressure. This keeps the reference anchors
    and removes direct mean-stress feedback from the damage resistance.
    """

    network: MatrixEvolutionNetwork = field(kw_only=True)
    resistance_uses_pressure: bool = field(default=True, kw_only=True)

    def evolution(self, d, effective_mean, fabric_projection=0.):
        resistance, resistance_d = self.resistance(d)
        beta, beta_d = self.dilatancy(d)
        mu, mu_d = self.friction(d)
        g, gd, gp, gf = self.network.value_derivatives(d, effective_mean, fabric_projection)
        if not self.resistance_uses_pressure:
            reference = self.network.value_derivatives(d, self.network.pressure_scale,
                                                       fabric_projection)
            g[0], gd[0], gp[0], gf[0] = reference[0][0], reference[1][0], 0., reference[3][0]
        factors = np.exp(g)
        value = resistance*factors[0]
        gap = (mu-beta)*factors[1]
        return (value, factors[0]*(resistance_d+resistance*gd[0]), value*gp[0],
                mu-gap, mu_d-factors[1]*(mu_d-beta_d+(mu-beta)*gd[1]), -gap*gp[1],
                value*gf[0], -gap*gf[1])
