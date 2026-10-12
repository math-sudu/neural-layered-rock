"""History-aware neural plane storage inside the saturated material energy.

The trainable tanh network learns a monotone damage-to-compliance relation.
Stress, pressure, damage release and plane backstress retain the derivatives
of the same energy. The implicit layer supplies contact and evolution laws.
This is a physics-constrained neural constitutive model, not a PDE PINN.
"""

from dataclasses import dataclass

import numpy as np

from .poromechanical_coupling import PorousEnergyClosure


@dataclass(frozen=True)
class MonotoneStorageNetwork:
    """One input, a tanh hidden layer and a positive linear output with skip.

    For x=(d-d0)/scale, f(x)=x+sum(v*(tanh(a*x+b)-tanh(b))).
    Positive a and v give f'>0. The physical output is
    h(d)=d0+scale*f(x)/f'(0), anchoring h(d0)=d0 and h'(d0)=1.
    Thus learning changes the curvature of the storage law, rather than
    relabelling its reference compliance or the initial plane density.
    """

    raw_parameters: np.ndarray
    initial_density: float
    density_scale: float

    def __post_init__(self):
        raw = np.asarray(self.raw_parameters, dtype=float)
        if raw.ndim != 1 or not raw.size or raw.size % 3 or not np.isfinite(raw).all():
            raise ValueError("Network parameters must contain finite weight/slope/bias triples.")
        if self.initial_density <= 0 or self.density_scale <= 0:
            raise ValueError("Positive reference density and density scale are required.")
        object.__setattr__(self, "raw_parameters", raw.copy())

    @classmethod
    def initialize(cls, initial_density, density_scale, width=4):
        if width < 1:
            raise ValueError("The hidden layer must have at least one neuron.")
        raw = np.r_[np.full(width, np.log(np.expm1(.05))),
                    np.full(width, np.log(np.expm1(1.))),
                    np.linspace(-.5, -2., width)]
        return cls(raw, initial_density, density_scale)

    def value_derivatives(self, density):
        """Return h, dh/dd and d2h/dd2 for scalars or arrays of densities."""
        d = np.asarray(density)
        raw_v, raw_a, b = np.split(self.raw_parameters, 3)
        v, a = np.logaddexp(0., raw_v), np.logaddexp(0., raw_a)
        x = (d-self.initial_density)/self.density_scale
        t = np.tanh(x[..., None]*a+b)
        t0 = np.tanh(b)
        normalization = 1.+np.sum(v*a*(1.-t0*t0))
        f = x+np.sum(v*(t-t0), axis=-1)
        fp = 1.+np.sum(v*a*(1.-t*t), axis=-1)
        fpp = -2.*np.sum(v*a*a*t*(1.-t*t), axis=-1)
        return (self.initial_density+self.density_scale*f/normalization,
                fp/normalization, fpp/(normalization*self.density_scale))


class NeuralPorousEnergy(PorousEnergyClosure):
    """Use H(d)=H0/(1+alpha*h_theta(d)) for finite-plane storage.

    Matrix compliance, constituent storage, Biot tensor and the distinct
    matrix/plane history variables keep their physical definitions. The
    network replaces the linear localization dependence on matrix damage.
    It receives no specimen identifier, measured pressure or curve index.
    """

    def __init__(self, elastic, normal, *, network, matrix_slope, **kwargs):
        super().__init__(elastic, normal, matrix_slope=matrix_slope, **kwargs)
        if self.plane_reference_damage is not None:
            raise ValueError("Neural storage requires an evolving plane energy.")
        self.network = network
        self.matrix_slope = matrix_slope
        self.plane_H0 = np.linalg.inv(self.plane_S0)

    def plane_storage_derivatives(self, d):
        h, hp, hpp = self.network.value_derivatives(d)
        a = self.matrix_slope
        s = 1.+a*h
        if s <= 0:
            raise ValueError("Neural plane compliance is nonpositive outside the damage domain.")
        return (self.plane_H0/s,
                -self.plane_H0*(a*hp/s**2),
                self.plane_H0*(2.*a*a*hp*hp/s**3-a*hpp/s**2))

    def plane_storage(self, d):
        H, Hp, _ = self.plane_storage_derivatives(d)
        return H, Hp
