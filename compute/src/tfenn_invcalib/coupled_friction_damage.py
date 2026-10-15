"""Implicit matrix/plane friction and damage in the TI porous energy.

Compression is positive. Matrix flow is a TI quadratic deviatoric potential
with independent dilation; flat unbonded planes obey Coulomb contact. All
forces, pore pressure and damage release are evaluated at the same end state.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, least_squares

from .energetic_coupling import KDEV, VOLUME


class ConstitutiveSolveError(RuntimeError):
    """The requested material state has no accepted numerical solution."""


def _resolve_candidate(solution, fun, jac):
    """Check state convergence as well as the scaled constitutive residual.

    Near a fold, a nonzero least-squares minimum can have a small residual
    and gradient without being a root. Refine nominal candidates with local
    Newton corrections, using the state tolerance of the nonlinear solve.
    """
    x = solution.x.copy()
    residual = fun(x)
    if not np.all(np.isfinite(residual)) or max(abs(residual)) > 1e-7:
        return False
    for _ in range(12):
        norm = np.linalg.norm(residual)
        if norm <= len(x)*np.finfo(float).eps:
            solution.x, solution.fun = x, residual
            return True
        try:
            tangent = jac(x)
            correction, _, rank, _ = np.linalg.lstsq(tangent, -residual, rcond=None)
        except np.linalg.LinAlgError:
            return False
        # At zero matrix flow and zero mean force the volume-flow fraction
        # is free. Check the determined state with its minimum correction;
        # other rank losses can indicate a branch fold.
        free_fraction = rank == len(x)-1 and not np.any(tangent[:, -1])
        if rank < len(x) and not free_fraction:
            return False
        linear_error = residual+tangent @ correction
        rounding = len(x)*np.finfo(float).eps*(1+norm+np.linalg.norm(tangent)*np.linalg.norm(correction))
        if not free_fraction and np.linalg.norm(linear_error) > rounding:
            return False
        size = 1+np.linalg.norm(x)
        if np.linalg.norm(correction) <= 1e-10*size:
            solution.x, solution.fun = x, residual
            return True
        if not np.all(np.isfinite(correction)) or np.linalg.norm(correction) > size:
            return False
        fraction = 1.
        while fraction*np.linalg.norm(correction) > 1e-10*size:
            trial = x+fraction*correction
            trial_residual = fun(trial)
            if np.linalg.norm(trial_residual) < norm:
                x, residual = trial, trial_residual
                break
            fraction *= .5
        else:
            return False
    return False


@dataclass(frozen=True)
class FrictionDamageLaw:
    matrix_cohesion: float
    matrix_friction: float
    matrix_dilatancy: float
    parallel_weight: float
    shear_weight: float
    plane_friction: float
    damage_resistance: float
    damage_peak_density: float | None = None
    damage_residual_fraction: float = 0.
    damage_shape: float = 1.
    dilation_residual_fraction: float = 1.
    parallel_residual_weight: float | None = None
    dilation_density_ratio: float = 1.
    dilation_initial_density: float | None = None
    matrix_contraction: float = 0.
    friction_residual_fraction: float = 1.

    def __post_init__(self):
        if not (self.matrix_cohesion > 0 and self.damage_resistance > 0
                and min(self.parallel_weight, self.shear_weight) > 0
                and 0 <= self.matrix_dilatancy <= self.matrix_friction
                and self.plane_friction >= 0):
            raise ValueError("Positive strengths/weights and 0 <= dilation <= friction are required.")
        if (self.damage_peak_density is not None and self.damage_peak_density <= 0
                or not 0 <= self.damage_residual_fraction < 1 or self.damage_shape <= 0
                or not 0 <= self.dilation_residual_fraction <= 1
                or self.dilation_density_ratio <= 0
                or self.dilation_initial_density is not None and self.dilation_initial_density < 0
                or self.matrix_contraction < 0
                or self.matrix_contraction > 0 and (self.dilation_initial_density is None
                                                    or self.damage_peak_density is None)
                or not 0 <= self.friction_residual_fraction <= 1
                or self.friction_residual_fraction != 1 and self.damage_peak_density is None
                or self.parallel_residual_weight is not None and self.parallel_residual_weight <= 0):
            raise ValueError("Positive damage peak density/shape, 0 <= resistance fraction < 1, "
                             "positive flow scales/weights and 0 <= dilation fraction <= 1 are required. "
                             "Shear contraction needs a nonnegative amplitude and a density origin/scale; "
                             "friction evolution needs a peak density and a residual fraction in [0, 1].")

    def friction_scale(self, d):
        """Peak-to-residual friction using the existing resistance-peak density."""
        if self.damage_peak_density is None or d <= self.damage_peak_density:
            return 1., 0.
        transient = (1-self.friction_residual_fraction)*self.damage_peak_density/d
        return self.friction_residual_fraction+transient, -transient/d

    def friction(self, d):
        scale, derivative = self.friction_scale(d)
        return self.matrix_friction*scale, self.matrix_friction*derivative

    def parallel_metric_weight(self, d):
        """Relax the directional weight on the descending damage branch.

        The convex transition r*(2-r), r=d_peak/d, has zero slope at its
        onset and approaches the positive residual weight. Damage resistance
        and strength directionality have distinct roles; neither residual
        isotropy nor permanent peak anisotropy is imposed on the material fit.
        """
        if (self.parallel_residual_weight is None or self.damage_peak_density is None
                or d <= self.damage_peak_density):
            return self.parallel_weight, 0.
        difference = self.parallel_weight-self.parallel_residual_weight
        ratio = self.damage_peak_density/d
        return (self.parallel_residual_weight+difference*ratio*(2-ratio),
                -2*difference*ratio*(1-ratio)/d)

    def resistance(self, d):
        """Positive resistance and derivative in the matrix-density coordinate.

        A missing peak density selects the constant-resistance limit. Otherwise
        the rational rise/fall generalizes Ren et al. (2025), Eq. (16): an
        unshifted density ratio, adjustable exponent and optional residual floor.
        """
        if self.damage_peak_density is None:
            return self.damage_resistance, 0.
        x = (d/self.damage_peak_density)**self.damage_shape
        r = self.damage_residual_fraction
        value = self.damage_resistance*(r+(1-r)*2*x/(1+x*x))
        derivative = (self.damage_resistance*(1-r)*2*self.damage_shape*x*(1-x*x)
                      / (d*(1+x*x)**2))
        return value, derivative

    def dilatancy(self, d):
        if self.damage_peak_density is None:
            return self.matrix_dilatancy, 0.
        scale = self.damage_peak_density*self.dilation_density_ratio
        r = self.dilation_residual_fraction
        friction_scale, friction_prime = self.friction_scale(d)
        amplitude = self.matrix_dilatancy*friction_scale
        amplitude_prime = self.matrix_dilatancy*friction_prime
        if self.dilation_initial_density is not None:
            # Distributed shear can compact before dilation mobilizes. Its
            # contraction term decays as the existing damage density grows.
            # beta <= friction(d) still gives nonnegative sliding work;
            # that condition does not require beta to be nonnegative.
            x = max(0., d-self.dilation_initial_density)/scale
            value = 4*x/(1+x)**2
            derivative = (4*(1-x)/(scale*(1+x)**3)
                          if d >= self.dilation_initial_density else 0.)
            contraction = self.matrix_contraction/(1+x)**2
            contraction_prime = (-2*self.matrix_contraction/(scale*(1+x)**3)
                                 if d >= self.dilation_initial_density else 0.)
            shape = r+(1-r)*value
            return (amplitude*shape-contraction,
                    amplitude_prime*shape+amplitude*(1-r)*derivative-contraction_prime)
        x = d/scale
        shape = r+(1-r)/(1+x)
        return (amplitude*shape,
                amplitude_prime*shape-amplitude*(1-r)/(scale*(1+x)**2))

    def evolution(self, d, effective_mean, fabric_projection=0.):
        """R, R_d, R_p, beta, beta_d, beta_p, R_fabric, beta_fabric."""
        resistance, resistance_d = self.resistance(d)
        beta, beta_d = self.dilatancy(d)
        return resistance, resistance_d, 0., beta, beta_d, 0., 0., 0.


def complementarity(a, b):
    """Fischer--Burmeister residual: a>=0, b>=0, a*b=0."""
    return np.hypot(a, b)-a-b


def complementarity_partials(a, b):
    radius = np.hypot(a, b)
    return (a/radius-1, b/radius-1) if radius > 0 else (-1., -1.)


class CoupledFrictionDamage:
    def __init__(self, energy, law, plane_density, *, fabric_source="plastic_stress"):
        if plane_density <= 0:
            raise ValueError("Plane density must be positive.")
        if fabric_source not in ("plastic_stress", "matrix_force"):
            raise ValueError("Fabric direction requires plastic_stress or matrix_force.")
        self.energy, self.law, self.D = energy, law, plane_density
        self.fabric_source = fabric_source
        self.fabric_projector = 1.5*(energy.B[:, 0]-VOLUME/3)
        normal_mode = np.sqrt(1.5)*(energy.B[:, 0]-VOLUME/3)
        normal_projector = np.outer(normal_mode, normal_mode)
        shear_projector = energy.B[:, 1:] @ energy.B[:, 1:].T
        parallel_projector = KDEV-normal_projector-shear_projector
        self.parallel_metric = 1.5*parallel_projector
        # Each projector is orthogonal. Equal weights recover von Mises q.
        self.metric = 1.5*(normal_projector + law.parallel_weight*parallel_projector
                          + law.shear_weight*shear_projector)

    def matrix_metric(self, d):
        if d is None:
            return self.metric, np.zeros((6, 6))
        weight, derivative = self.law.parallel_metric_weight(d)
        return (self.metric+(weight-self.law.parallel_weight)*self.parallel_metric,
                derivative*self.parallel_metric)

    def fabric_direction(self, stress):
        """Objective direction projection and its stress gradient.

        Applied to effective stress with positive triaxial q, this is
        (3 sin(theta)^2-1)/2. Applied to matrix force, its direction also
        evolves with the energetic backstress. A hydrostatic input has no direction;
        its value and gradient are set to zero, as for the matrix flow norm.
        """
        deviator = stress-VOLUME*(VOLUME @ stress)/3
        q = np.sqrt(1.5*(deviator @ deviator))
        if q == 0:
            return 0., np.zeros(6)
        value = (self.fabric_projector @ deviator)/q
        return value, (self.fabric_projector-value*1.5*deviator/q)/q

    def matrix_surface(self, force, d=None, *, effective_mean=None, flow_fraction=None,
                       fabric_projection=0.):
        deviator = KDEV @ force
        metric, _ = self.matrix_metric(d)
        mapped = metric @ deviator
        q = np.sqrt(max(0., deviator @ mapped))
        mean = force @ VOLUME/3
        direction = mapped/q if q > 0 else np.zeros(6)
        beta = (self.law.matrix_dilatancy if d is None
                else self.law.evolution(d, effective_mean, fabric_projection)[3])
        fraction = float(mean > 0) if flow_fraction is None else flow_fraction
        direction = direction-fraction*beta*VOLUME/3
        friction = self.law.matrix_friction if d is None else self.law.friction(d)[0]
        strength = self.law.matrix_cohesion+friction*max(0., mean)
        return q-strength, direction, q, mean

    def consolidated_state(self, effective_confinement, d, pressure=0.):
        """Drained isotropic consolidation, including any attained cap strain."""
        if effective_confinement < 0 or d <= 0:
            raise ValueError("Nonnegative confinement and positive matrix density are required.")
        m = self.energy
        total = (effective_confinement+pressure)*VOLUME
        kappa = max(0., (effective_confinement-m.compaction_onset)/m.compaction_hardening)
        strain = m.S0 @ (total-m.biot*pressure)+kappa*m.compaction_mode
        cm, z = np.zeros(6), np.zeros(3)
        zeta = m.fluid_content(strain, cm, z, kappa, pressure)
        return dict(strain=strain, matrix_strain=cm, plane_modes=z, compaction=kappa,
                    d=d, fluid_content=zeta,
                    response=m.response(strain, cm, z, kappa, d, self.D, zeta))

    def _mixed_kinematics(self, axial_strain, transverse_stress, pressure, fluid_content):
        """Eliminate the five free strains exactly, for any inelastic strain."""
        m = self.energy
        if pressure is None:
            stiffness = m.C0+m.M*np.outer(m.biot, m.biot)
            coupling = -m.C0+m.M*np.outer(m.biot, VOLUME-m.biot)
            offset = m.M*m.biot*fluid_content
        else:
            stiffness, coupling, offset = m.C0, -m.C0, m.biot*pressure
        base = np.r_[axial_strain, np.linalg.solve(
            stiffness[1:, 1:], np.asarray(transverse_stress)
            -stiffness[1:, 0]*axial_strain-offset[1:])]
        transform = np.vstack((np.zeros(6), np.linalg.solve(
            stiffness[1:, 1:], -coupling[1:, :])))
        return base, transform

    def step(self, axial_strain, transverse_stress, previous, *,
             pressure=None, fluid_content=None, initial_guess=None,
             density_search_subdivisions=1):
        """Backward-Euler mixed-control solve of all five active processes.

        Unknowns are matrix strain, both tangential slips, normal opening,
        compaction, log(d/d_previous), the matrix friction multiplier and the
        volume-flow fraction at the zero-mean-force corner. Normal
        contact, cap, damage and friction consistency use complementarity;
        total tractions and the fluid constraint are eliminated algebraically.
        Plane flow uses the equivalent proximal Coulomb equation, which remains
        defined at open contact where tangential force vanishes during slip.
        An optional initial_guess supplies a solver seed while previous remains
        the history origin; this supports comparing roots at the same loading.
        density_search_subdivisions refines the doubling grid used to bracket
        a damage-consistency root when the preceding candidates are unresolved.
        A nonzero constitutive residual raises rather than returning a state.
        """
        if (pressure is None) == (fluid_content is None):
            raise ValueError("Specify exactly one fluid boundary condition.")
        if density_search_subdivisions < 1:
            raise ValueError("At least one density-search subdivision is required.")
        m, law = self.energy, self.law
        base, transform = self._mixed_kinematics(
            axial_strain, transverse_stress, pressure, fluid_content)
        cm0, z0, k0, d0 = (previous["matrix_strain"], previous["plane_modes"],
                           previous["compaction"], previous["d"])
        strain_scale = 1e-3
        stress_scale = np.linalg.norm(m.C0, ord=2)*strain_scale
        identity = np.eye(13)
        dcm = strain_scale*identity[:6]
        dz = strain_scale*identity[[8, 6, 7]]
        dk = strain_scale*identity[9]
        dei = dcm+m.B @ dz+np.outer(m.compaction_mode, dk)
        dee = (transform-np.eye(6)) @ dei
        du = (m.M*(m.biot @ dee+VOLUME @ dei) if pressure is None else np.zeros(13))
        dtau = m.C0 @ dee+np.outer(m.biot-VOLUME, du)
        dp_effective = VOLUME @ dtau/3

        def evaluate(x, damage_active=False, matrix_corner=False, matrix_active=False, plane_stick=False):
            cm = x[:6]*strain_scale
            z = np.r_[x[8], x[6:8]]*strain_scale
            kappa, d = x[9]*strain_scale, d0*np.exp(x[10])
            lm = x[11]*strain_scale
            ei = m.inelastic_strain(cm, z, kappa)
            strain = base+transform @ ei
            zeta = (fluid_content if pressure is None else
                    m.fluid_content(strain, cm, z, kappa, pressure))
            f = m.response(strain, cm, z, kappa, d, self.D, zeta)
            effective_mean = VOLUME @ f["plastic_stress"]/3
            fabric_projection, fabric_gradient = self.fabric_direction(f[self.fabric_source])
            fm, flow, q, mean = self.matrix_surface(f["matrix_force"], d,
                                                   effective_mean=effective_mean,
                                                   flow_fraction=x[12], fabric_projection=fabric_projection)
            tangent = f["plane_force"][1:]
            norm = np.linalg.norm(tangent)
            reaction = f["plane_force"][0]
            # Mandel traction has magnitude sqrt(2)*physical shear traction.
            friction = np.sqrt(2)*law.plane_friction*max(0., reaction)
            fp = norm-friction
            slip = (z[1:]-z0[1:])/strain_scale
            proximal = slip+tangent/stress_scale
            radius = np.linalg.norm(proximal)
            threshold = friction/stress_scale
            shrink = (1-threshold/radius)*proximal if radius > threshold else np.zeros(2)
            (resistance, resistance_d, resistance_p, beta, beta_d, beta_p,
             resistance_fabric, beta_fabric) = law.evolution(d, effective_mean, fabric_projection)
            residual = np.r_[
                (cm-cm0-lm*flow)/strain_scale,
                slip-shrink,
                complementarity(-x[8], reaction/stress_scale),
                complementarity((kappa-k0)/strain_scale,
                                (m.compaction_onset-f["compaction_force"])/stress_scale),
                complementarity(x[10], 1-f["matrix_release"]/resistance),
                complementarity(x[11], -fm/stress_scale),
                x[12]-np.clip(x[12]+mean/stress_scale, 0., 1.),
            ]
            state = dict(strain=strain, matrix_strain=cm, plane_modes=z, compaction=kappa,
                         d=d, fluid_content=zeta, response=f,
                         matrix_multiplier=lm, plane_multiplier=np.linalg.norm(z[1:]-z0[1:]),
                         matrix_yield=fm, plane_yield=fp,
                         matrix_equivalent_force=q, matrix_mean_force=mean,
                         damage_resistance=resistance, matrix_dilatancy=beta,
                         matrix_flow_fraction=x[12],
                         fabric_projection=fabric_projection,
                         effective_mean_stress=effective_mean)

            dd = d*identity[10]
            H, Hp, Hpp = m.plane_storage_derivatives(d)
            dmatrix = dtau-m.Hc @ dcm/d+np.outer(m.Hc @ cm/d**2, dd)
            dfabric = fabric_gradient @ (dmatrix if self.fabric_source == "matrix_force" else dtau)
            dplane = m.B.T @ dtau-H @ dz/self.D-np.outer(Hp @ z/self.D, dd)
            metric, metric_prime = self.matrix_metric(d)
            force = f["matrix_force"]
            mapped = metric @ force
            dmapped = metric @ dmatrix+np.outer(metric_prime @ force, dd)
            dq = ((mapped @ dmatrix+.5*(force @ metric_prime @ force)*dd)/q
                  if q > 0 else np.zeros(13))
            dflow = (dmapped/q-np.outer(mapped, dq)/q**2
                     if q > 0 else np.zeros((6, 13)))
            dflow -= np.outer(VOLUME/3, x[12]*(beta_d*dd+beta_p*dp_effective+beta_fabric*dfabric)
                             +beta*identity[12])
            mu, mu_prime = law.friction(d)
            dfm = dq-mu*(VOLUME @ dmatrix)/3*(mean > 0)-mu_prime*max(0., mean)*dd
            dy = (cm @ m.Hc @ dcm/d**2-2*f["matrix_self_release"]*dd/d
                  -z @ Hp @ dz/self.D-.5*(z @ Hpp @ z)*dd/self.D)
            jacobian = np.zeros((13, 13))
            jacobian[:6] = (dcm-lm*dflow)/strain_scale-np.outer(flow, identity[11])
            dslip = dz[1:]/strain_scale
            jacobian[6:8] = dslip
            if plane_stick:
                residual[6:8] = slip
            elif radius > threshold:
                dproximal = dslip+dplane[1:]/stress_scale
                dthreshold = np.sqrt(2)*law.plane_friction*dplane[0]*(reaction > 0)/stress_scale
                shrink_jacobian = ((1-threshold/radius)*np.eye(2)
                                  +threshold*np.outer(proximal, proximal)/radius**3)
                jacobian[6:8] -= (shrink_jacobian @ dproximal
                                   -np.outer(proximal/radius, dthreshold))
            pairs = [
                (-x[8], reaction/stress_scale, -identity[8], dplane[0]/stress_scale),
                ((kappa-k0)/strain_scale, (m.compaction_onset-f["compaction_force"])/stress_scale,
                 identity[9], -(m.compaction_mode @ dtau-m.compaction_hardening*dk)/stress_scale),
                (x[10], 1-f["matrix_release"]/resistance, identity[10],
                 -dy/resistance+f["matrix_release"]*(resistance_d*dd
                    +resistance_p*dp_effective+resistance_fabric*dfabric)/resistance**2),
                (x[11], -fm/stress_scale, identity[11], -dfm/stress_scale),
            ]
            for row, (a, b, da, db) in enumerate(pairs, start=8):
                pa, pb = complementarity_partials(a, b)
                jacobian[row] = pa*da+pb*db
                if (row == 10 and damage_active) or (row == 11 and matrix_active):
                    residual[row], jacobian[row] = b, db
            # The subgradient of <p_c>_+ is 0 below zero, 1 above zero,
            # and [0, 1] at the corner. Its projection equation retains the
            # admissible intermediate volume flow without a smoothing scale.
            jacobian[12] = identity[12]
            if 0 < x[12]+mean/stress_scale < 1:
                jacobian[12] -= identity[12]+(VOLUME @ dmatrix)/(3*stress_scale)
            if matrix_corner:
                residual[12] = mean/stress_scale
                jacobian[12] = (VOLUME @ dmatrix)/(3*stress_scale)
            return residual, state, jacobian

        cached_x, cached_value, cached_branches = None, None, (False, False, False, False)

        def cached(x, damage_active=False, matrix_corner=False, matrix_active=False, plane_stick=False):
            nonlocal cached_x, cached_value, cached_branches
            branches = (damage_active, matrix_corner, matrix_active, plane_stick)
            if (cached_x is None or not np.array_equal(x, cached_x)
                    or branches != cached_branches):
                cached_x, cached_value = x.copy(), evaluate(x, *branches)
                cached_branches = branches
            return cached_value

        guess = previous if initial_guess is None else initial_guess
        initial = np.r_[guess["matrix_strain"]/strain_scale,
                        guess["plane_modes"][1:]/strain_scale, guess["plane_modes"][0]/strain_scale,
                        guess["compaction"]/strain_scale, np.log(guess["d"]/d0),
                        0. if initial_guess is None else guess.get("matrix_multiplier", 0.)/strain_scale,
                        guess.get("matrix_flow_fraction", 1.)]
        solution = least_squares(lambda x: cached(x)[0], initial, jac=lambda x: cached(x)[2],
                                 x_scale="jac", ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=200)
        resolved = _resolve_candidate(solution, lambda x: cached(x)[0], lambda x: cached(x)[2])
        residual, state, _ = cached(solution.x)
        solver_route = "previous_state" if initial_guess is None else "supplied_guess"
        if not resolved:
            # Near first activation the complementarity least-squares problem
            # can stall at a nonzero local minimum with almost no damage. Solve
            # the active equation Y=R directly, then require all original
            # complementarity equations, including damage irreversibility.
            active_seed = solution.x.copy()
            active_seed[10] = max(0., active_seed[10])
            lower = np.full(13, -np.inf)
            lower[10] = 0.
            active = least_squares(lambda x: cached(x, True)[0], active_seed,
                jac=lambda x: cached(x, True)[2], x_scale="jac",
                bounds=(lower, np.full(13, np.inf)),
                ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=200)
            resolved = _resolve_candidate(active, lambda x: cached(x)[0], lambda x: cached(x)[2])
            active_residual, active_state, _ = cached(active.x)
            if resolved:
                solution, residual, state = active, active_residual, active_state
                solver_route = "active_damage"
        if not resolved:
            # The projection equation can stall while approaching p_c=0 from
            # full contraction. Solve that exact corner branch, then check the
            # original equations, which also require the fraction in [0, 1].
            # Enforce both the fraction and friction-multiplier bounds. A
            # negative multiplier can otherwise trap the corner solve on an
            # inadmissible unloading branch when contraction changes rapidly.
            corner_lower, corner_upper = np.full(13, -np.inf), np.full(13, np.inf)
            corner_lower[11] = 0.
            corner_lower[12], corner_upper[12] = 0., 1.
            corner_seed = solution.x.copy()
            corner_seed[11] = max(0., corner_seed[11])
            corner_seed[12] = .5
            corner = least_squares(lambda x: cached(x, matrix_corner=True)[0], corner_seed,
                jac=lambda x: cached(x, matrix_corner=True)[2], x_scale="jac",
                bounds=(corner_lower, corner_upper),
                ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=200)
            resolved = _resolve_candidate(corner, lambda x: cached(x)[0], lambda x: cached(x)[2])
            corner_residual, corner_state, _ = cached(corner.x)
            if resolved:
                solution, residual, state = corner, corner_residual, corner_state
                solver_route = "matrix_corner"
            else:
                # A bounded corner solve can reach full volume flow while the
                # admissible damage jump lies just beyond p_c=0. Its coupled
                # strain/damage state supplies a seed for that adjacent branch;
                # the original projection and complementarity equations decide
                # acceptance, without imposing the corner or a damage jump.
                released = least_squares(lambda x: cached(x)[0], corner.x,
                    jac=lambda x: cached(x)[2], x_scale="jac",
                    ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=200)
                resolved = _resolve_candidate(released, lambda x: cached(x)[0], lambda x: cached(x)[2])
                released_residual, released_state, _ = cached(released.x)
                if resolved:
                    solution, residual, state = released, released_residual, released_state
                    solver_route = "released_corner"
        if not resolved and law.damage_peak_density is not None:
            # Unresolved candidates trigger a search at increasing density.
            # Solve the remaining equations at each fixed density, then take
            # the first sampled positive-to-nonpositive consistency bracket.
            fixed_states = {}

            def fixed_density(log_ratio):
                if log_ratio in fixed_states:
                    candidate = fixed_states[log_ratio]
                else:
                    seed = (fixed_states[min(fixed_states, key=lambda v: abs(v-log_ratio))].x.copy()
                            if fixed_states else initial.copy())
                    seed[10] = log_ratio

                    def fixed_residual(x):
                        result = cached(x)[0].copy()
                        result[10] = x[10]-log_ratio
                        return result

                    def fixed_jacobian(x):
                        result = cached(x)[2].copy()
                        result[10] = identity[10]
                        return result

                    candidate = least_squares(fixed_residual, seed, jac=fixed_jacobian,
                        x_scale="jac", ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=200)
                    if not _resolve_candidate(candidate, fixed_residual, fixed_jacobian):
                        raise ConstitutiveSolveError(f"Fixed-density constitutive solve failed at "
                                           f"axial strain {axial_strain:g}, log(d/d_previous)={log_ratio:g}.")
                    fixed_states[log_ratio] = candidate
                result = cached(candidate.x)[1]
                return result["response"]["matrix_release"]/result["damage_resistance"]-1

            try:
                lower = 0.
                if fixed_density(lower) <= 0:
                    log_ratio = lower
                else:
                    for _ in range(40*density_search_subdivisions):
                        upper = lower+np.log(2.)/density_search_subdivisions
                        if fixed_density(upper) <= 0:
                            log_ratio = brentq(fixed_density, lower, upper, xtol=1e-11)
                            fixed_density(log_ratio)
                            break
                        lower = upper
                    else:
                        raise ConstitutiveSolveError(f"Damage consistency root was not bracketed at axial strain {axial_strain:g}.")
                solution = fixed_states[log_ratio]
                resolved = _resolve_candidate(solution, lambda x: cached(x)[0], lambda x: cached(x)[2])
                solver_route = "fixed_density"
                if not resolved:
                    # Multiple states can solve the fixed-density subproblem.
                    # A switch between them can close a scalar sign bracket
                    # without satisfying damage consistency. Release that
                    # endpoint to the full system before trying other seeds.
                    solution = least_squares(lambda x: cached(x)[0], solution.x,
                        jac=lambda x: cached(x)[2], x_scale="jac",
                        ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=200)
                    resolved = _resolve_candidate(solution, lambda x: cached(x)[0], lambda x: cached(x)[2])
                    solver_route = "released_fixed_density"
                if not resolved:
                    raise ConstitutiveSolveError(f"Fixed-density root did not resolve the coupled state "
                                                 f"at axial strain {axial_strain:g}.")
                residual, state, _ = cached(solution.x)
            except ConstitutiveSolveError:
                # A failed fixed-density subproblem does not exclude a coupled
                # damage root. The solve can stall near first matrix activation
                # while the plane is already sliding. Resolve matrix friction
                # and damage consistency with the plane temporarily fixed,
                # then release that predictor to the original coupled system.
                # The predictor itself can violate the plane yield surface;
                # only the released, fully admissible state may be accepted.
                predictor_lower = corner_lower.copy()
                predictor_lower[10] = 0.
                # Preserve the corner-seeded branch when it succeeds. If it
                # fails, use the history seed: the failed zero-mean corner
                # may be far from a positive-mean activation root.
                for seed in (corner.x, initial):
                    predictor_seed = np.clip(seed, predictor_lower, corner_upper)
                    predictor = least_squares(lambda x: cached(x, True, False, True, True)[0],
                        predictor_seed, jac=lambda x: cached(x, True, False, True, True)[2],
                        bounds=(predictor_lower, corner_upper), x_scale="jac",
                        ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=200)
                    advanced = least_squares(lambda x: cached(x)[0], predictor.x,
                        jac=lambda x: cached(x)[2], x_scale="jac",
                        ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=200)
                    resolved = _resolve_candidate(advanced, lambda x: cached(x)[0], lambda x: cached(x)[2])
                    advanced_residual, advanced_state, _ = cached(advanced.x)
                    if resolved:
                        solver_route = "released_plane_predictor"
                        break
                else:
                    # At first activation the volume fraction can have nearly
                    # zero Jacobian sensitivity. Its dynamic scaling can stall
                    # the corner search. Reuse the physical strain/log-density
                    # scales, then release this predictor to the full equations.
                    scaled_corner = least_squares(lambda x: cached(x, matrix_corner=True)[0],
                        corner_seed, jac=lambda x: cached(x, matrix_corner=True)[2],
                        bounds=(corner_lower, corner_upper), x_scale=1.,
                        ftol=1e-10, xtol=1e-10, gtol=1e-10)
                    advanced = least_squares(lambda x: cached(x)[0], scaled_corner.x,
                        jac=lambda x: cached(x)[2], x_scale="jac",
                        ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=200)
                    resolved = _resolve_candidate(advanced, lambda x: cached(x)[0], lambda x: cached(x)[2])
                    if not resolved:
                        # Dynamic scaling can stall the active matrix/held-plane
                        # predictor while the zero-mean
                        # corner search also stalls. Keep its physical scales
                        # and release it before testing the full equations.
                        predictor_seed = np.clip(corner.x, predictor_lower, corner_upper)
                        predictor_cost = np.inf
                        while True:
                            scaled_predictor = least_squares(
                                lambda x: cached(x, True, False, True, True)[0], predictor_seed,
                                jac=lambda x: cached(x, True, False, True, True)[2],
                                bounds=(predictor_lower, corner_upper), x_scale=1.,
                                ftol=1e-10, xtol=1e-10, gtol=1e-10)
                            advanced = least_squares(lambda x: cached(x)[0], scaled_predictor.x,
                                jac=lambda x: cached(x)[2], x_scale="jac",
                                ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=200)
                            resolved = _resolve_candidate(advanced, lambda x: cached(x)[0], lambda x: cached(x)[2])
                            # A call's evaluation budget can end before this
                            # activation predictor supplies a usable full-system
                            # seed. Continue a decreasing predictor only after
                            # its released candidate has failed acceptance.
                            if (resolved or scaled_predictor.status != 0
                                    or scaled_predictor.cost >= predictor_cost):
                                break
                            predictor_cost = scaled_predictor.cost
                            predictor_seed = scaled_predictor.x
                        if not resolved:
                            # A near-fold local minimum may trap all predictors
                            # close to the preceding density. Release a higher
                            # density history seed to the unchanged equations.
                            density_seed = initial.copy()
                            density_seed[10] = np.log(2.)
                            density_seed[11] = previous.get("matrix_multiplier", 0.)/strain_scale
                            advanced = least_squares(lambda x: cached(x)[0], density_seed,
                                jac=lambda x: cached(x)[2], x_scale="jac",
                                ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=200)
                            resolved = _resolve_candidate(advanced, lambda x: cached(x)[0], lambda x: cached(x)[2])
                            if not resolved:
                                raise
                            solver_route = "higher_density_history"
                        else:
                            solver_route = "released_scaled_plane_predictor"
                    else:
                        solver_route = "released_scaled_corner"
                    advanced_residual, advanced_state, _ = cached(advanced.x)
                solution, residual, state = advanced, advanced_residual, advanced_state
        # The optimizer flag describes its own stopping rule. A reached
        # iteration limit can still supply a root of the constitutive equations;
        # conversely, a small optimization step need not supply one.
        if not resolved:
            raise ConstitutiveSolveError(f"Coupled constitutive solve failed at axial strain {axial_strain:g}: "
                               f"{solution.message}; scaled residual={max(abs(residual)):.3g}")
        f = state["response"]
        state["solver_route"] = solver_route
        state["dissipation_increments"] = {
            "matrix_sliding": f["matrix_force"] @ (state["matrix_strain"]-cm0),
            "plane_sliding": f["plane_force"][1:] @ (state["plane_modes"][1:]-z0[1:]),
            "normal_contact": f["plane_force"][0]*(state["plane_modes"][0]-z0[0]),
            "compaction": f["compaction_force"]*(state["compaction"]-k0),
            "damage": f["matrix_release"]*(state["d"]-d0),
        }
        return state
