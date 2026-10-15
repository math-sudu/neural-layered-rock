# Disentangling deformation and strengthening mechanisms in saturated layered rock with a friction–damage-informed thermodynamic neural network

Reproduction materials for the manuscript by Pengcheng Zhu and Tielin Chen,
prepared for *Acta Geotechnica*. Corresponding author: Tielin Chen,
Beijing Jiaotong University, Beijing 100044, China; tlchen1@bjtu.edu.cn.

The friction–damage-informed thermodynamic neural network distinguishes the
contributions of matrix deformation and bedding slip to axial strain,
effective stress and undrained strengthening. A common saturated free energy
and fluid balance couple analytical plane storage, `h(d)=d`, with a shared
matrix-evolution network. Its output mappings preserve non-negative
dissipation, while pressure histories constrain the volumetric evolution.
The network inputs are log damage density, calculated mean
effective stress and the objective stress direction relative to bedding.
The four hidden units supply resistance and signed volume flow. Both the
stress-only and joint models contain 22 material coordinates and 24 network
weights. Seven complete Winhausen histories enter parameter fitting; five
other orientations are excluded from those residuals. All twelve histories
informed development.

The selected full-precision inputs are `compute/parameters/stress.json` and
`compute/parameters/joint.json`. The repository also contains the source,
tests, original-data retrieval scripts, two warm starts and the lower-loss
joint control. The commands below reconstruct the complete histories and
their common-window and volume analysis in `outputs/`.

## Environment

Run the commands from the repository directory containing this README and
`compute/`.

Tested with CPython 3.13.7, NumPy 2.4.4 and SciPy 1.16.2 on Windows x86-64.
The constitutive calculations use double precision on the CPU.

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -r compute/requirements.txt
.venv/Scripts/python compute/scripts/fetch_winhausen_data.py
```

On POSIX use `.venv/bin/python`. Below, `python` denotes that interpreter.
For concurrent histories, limit numerical-library threads:

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
python -m pytest compute/tests -q
```

## Reproduce the reported histories

The full-precision `compute/parameters/stress.json` and `joint.json` define
the reported laws. Every reported history uses 200 time-distributed observation
stations plus early-loading and peak stations, a maximum axial increment of
7.8125e-6, a maximum logarithmic density increment of 0.05 and a subdivision
floor of 1e-9. Export every accepted step:

```powershell
python -B compute/scripts/trace_winhausen_mechanisms.py --parameters compute/parameters/stress.json --samples 200 --max-increment .0000078125 --max-log-density-increment .05 --min-increment .000000001 --workers 12 --output outputs/stress
python -B compute/scripts/trace_winhausen_mechanisms.py --parameters compute/parameters/joint.json --samples 200 --max-increment .0000078125 --max-log-density-increment .05 --min-increment .000000001 --workers 12 --output outputs/joint
python -B compute/scripts/compare_winhausen_refits.py --pair shared-evolution outputs/stress outputs/joint --output outputs/paired
python -B compute/scripts/analyze_winhausen_mechanisms.py --stress outputs/stress --joint outputs/joint --output outputs/paired
python -B compute/scripts/analyze_winhausen_strength.py --traces outputs/paired --pair outputs/stress outputs/joint
```

Use `--workers 1` for sequential execution. Each model yields twelve compressed
CSV histories and twelve JSON records with preparation states, coefficients,
integration settings and response metrics. The training and withheld sets have
1,510 and 1,083 observation stations per model. Maxima include all accepted
substeps; observed maxima use the complete retained acquisition record.

Expected withheld-history RMSEs in MPa:

| Specimen | Stress-only q | Stress-only pressure | Joint q | Joint pressure |
|---|---:|---:|---:|---:|
| OPA15 | 2.589 | 1.013 | 2.492 | 0.559 |
| OPA17 | 0.767 | 2.920 | 0.915 | 0.265 |
| OPA23 | 1.062 | 1.008 | 1.370 | 0.595 |
| OPA18 | 0.943 | 1.661 | 2.709 | 0.562 |
| OPA24 | 1.506 | 2.213 | 2.128 | 0.609 |

Path-balanced training/withheld pressure RMSEs are 2.030642/1.909219 MPa for
stress-only and 0.455922/0.533952 MPa for joint learning. Corresponding stress
RMSEs are 0.684430/1.521676 and 0.993662/2.039126 MPa. Each summary is the square
root of the mean of specimen mean-squared errors.

## Volume, mechanism and loading-stage analysis

`response_comparison.csv` contains 24 history summaries.
`observed_stage_comparison.csv` and `observed_stage_budgets.csv` each retain
48 windows, using first and last repeated observed pressure maxima and the
first observed stress maximum. Endpoints remain tied to original source rows,
with interpolation brackets recorded for the accepted-step histories.

`volume_coordinates.csv.gz` contains 272,598 original acquisition rows.
It retains observed and predicted volume coordinates under both each fit's own
coefficients and the joint model's common reference coefficients:
`v_ref = (delta_u - B_ref * delta_q) / M_ref`.
Own-coefficient predictions recover the total matrix, plane-opening and cap
volume changes since sealing. The stress-only OPA17 path opens late and OPA06
opens and recloses; these plane terms remain in the export. All selected joint
paths remain closed. Columns ending in `_increment` are changes between
successive acquisition rows, while the unsuffixed coordinates are cumulative.

`stage_partitions.csv` gives elastic, matrix and plane axial contributions
since sealing at the observed strength peak, both calculated peaks and the
endpoint. Plane shares use `plane_axial / (matrix_axial + plane_axial)`.
`activity_intervals.csv` records complete matrix flow, plane slip and signed
matrix-volume activity. Repeated imposed strains combine their increments;
current contraction can coexist with cumulative net dilation.

`strength_competition.csv.gz` resolves continuous closed-sliding changes into
slip storage, damage release and matrix-volume feedback. Steps that reach the
subdivision floor with a logarithmic density change above the adaptive limit
remain separate transition increments. Opening, non-sliding and symmetry
branches retain their actual stress changes without applying the sliding
identity. The observed-stage tables clip continuous edge steps to their
original endpoints and keep finite transitions whole.

`turning_events.csv`, `turning_intervals.csv` and
`dilation_supported_intervals.csv` locate model events independently of the
observed windows. The observed-stage budgets also separate mean effective
stress, actual plane reaction and matrix mean force. Conditional observed
increments `B_q*delta_q-delta_u` and `C_theta*delta_q+mu_p*delta_u` describe
volume-induced pressure reduction and closed-forward-sliding backstress,
respectively. Analytical angles distinguish the sign changes of `1/3-B_q`
and `sin(theta)^2-B_q` at each actual preparation stiffness.

OPA09's first observed pressure-to-strength window has plane axial share
0.656690, mean-effective-stress rise 0.454992 MPa and matrix-volume contribution
0.452951 MPa. Its net load increase is 0.223153 MPa, with slip, damage and
volume contributions +2.875034, -2.816525 and +0.164644 MPa.
OPA24's corresponding load increase is 1.041058 MPa: continuous sliding gives
1.191810 MPa and subsequent non-sliding increments give -0.150751 MPa.

## Data and loading conventions

The retrieval script downloads the twelve original files from
[Winhausen et al. (2023)](https://doi.org/10.1029/2023JB027382) and their
[OSF repository](https://osf.io/5s2cb/), preserving the bytes.
`compute/results/data/winhausen2023/source.json` records URLs, specimen
identities, angles, strain rates, preparation pressures, the seven/five split
and loading-segment exclusions. Cite the original experiment when reusing data.

The model angle is measured between bedding plane and loading axis, 90 degrees
minus the source angle. Raw strain percentages become dimensionless strains.
Stress and pressure are in MPa; compression is positive. Initial absolute
pressure defines zero excess pressure. Median reconstructed cell pressure
minus initial pore pressure gives effective preparation confinement.
The first q is applied as an elastic preload; subsequent loading is sealed.

Observation stations combine 200 uniform acquisition times, up to 16 early
rows through 0.2% axial strain, the observed q peak and both endpoints.
Duplicate selections merge. Pressure does not select them. A least-squares
monotone axial-strain estimate handles recording jitter while retaining source
rows and measured responses. OPA24 terminal unloading and OPA10's isolated
terminal pressure collapse at unchanged strain are excluded as specified in
`source.json`. The one recorded radial direction is not converted to volume.

The OPA10/OPA18 initial absolute pressure pairs differ from Table 3 of the
source article, while giving about 10 MPa effective confinement. Its pressure
maxima track absolute readings and its OPA15 stress-peak strain differs from
the raw file. This reproduction uses original files and recomputed increments
and peak strains, with exact boundaries recorded in `source.json`.

## Parameters and implementation

| Module | Role |
|---|---|
| `energetic_coupling.py` | Mandel geometry and common storage primitives |
| `poromechanical_coupling.py` | TI stiffness, saturated energy and conjugate forces |
| `coupled_friction_damage.py` | Implicit flow, contact, compaction and damage |
| `neural_storage.py` | Analytical identity storage and earlier neural closure |
| `neural_evolution.py` | Resistance and signed volume-flow network |
| `winhausen_data.py` | Original-data loader and specimen split |
| `winhausen_training.py` | Loading, sampling, material construction and residuals |

The 22 material coordinates and their bounds are defined in
`winhausen_training.py`. `cross_fraction` maps to the manuscript ratio as
`nu_pn = cross_fraction * sqrt((1-nu_parallel)*E_parallel/(2*E_normal))`;
`plane_compliance` is dimensionless plane density D and
`dilation_fraction` is beta_0/mu_0. Moduli are MPa. Fixed constituent,
localization and cap properties are in `fixed_parameters`.

The selected `storage_form="identity"` has zero storage-network weights.
Matrix weights are the row-major 4-by-3 input matrix, four biases and row-major
2-by-4 output matrix. Inputs are `log(d/d_peak)`, `p_eff/10-1` and
`(3/2)*(n.s_eff.n)/sqrt((3/2)*s_eff:s_eff)`, where
`s_eff=dev(Sigma-u*I)`. This direction becomes
`(3*sin(theta)^2-1)/2` in positive-q triaxial compression and zero at
hydrostatic input. The map is anchored at `d=d_peak, p_eff=10 MPa, chi=0`.
Calculated pressure and its derivatives enter the coupled implicit update.

The previous state seeds the nonlinear solve. Unresolved increments use
active-damage and matrix-corner reformulations, fixed-density continuation
and plane-holding predictors; acceptance always uses the full coupled
equations. At the matrix-force corner, the friction multiplier is nonnegative
and its volumetric subgradient lies in [0,1]. Regression inputs in
`compute/tests/data/` exercise the implemented activation and branch paths.

## Training and selection

Both final fits use seven histories, 200 base observation stations, maximum
axial increment 3.125e-5 and unit stress-defined stage-increment weight.
The objective combines equally weighted pre/post stress-peak mean squares,
channel maxima, responses at the observed stress peak and terminal pressure
when fitted. Additional response increments connect the half-peak-strain
station to the peak and the peak to its midpoint with the endpoint.
Both channels use the same stress-defined intervals.

`stress_start.json` is the preceding stress-only analytical fit. Four zero
direction weights embed it in the selected architecture. `joint_start.json`
already contains the 46-coordinate shared-evolution initialization.
Its two-feature resistance shift preserves the reference anchor and
large-density limit at fixed pressure and direction. Pressure targets enter
neither the stress-only initialization, fitting nor selection.

The final stress objective is 5.737251258513123. The retained joint candidate
is the first accepted common update, objective 31.937341781389414.
Continued fitting reaches 15.417805172726672 in `joint_control.json`, but
restores OPA24 softening over its observed pressure-to-strength windows.
The retained candidate has load growth and falling pressure over the principal
OPA09/12/22/24 windows. Selection considers that stage fidelity together with
complete response errors. Peak amplitude, timing and the short OPA15/17
pressure rises remain in the reported comparison.

Recreate the final warm starts and fitting batches in new output directories:

```powershell
python -B compute/scripts/train_winhausen_neural.py initialize --initialize-from compute/parameters/stress_start.json --channels stress --fabric-evolution --fabric-source plastic_stress --stage-increment-weight 1 --samples 200 --max-increment .00003125 --workers 7 --output outputs/stress-retrain
python -B compute/scripts/train_winhausen_neural.py train --channels stress --block joint --optimizer least-squares --trust-solver lsmr --coordinate-scaling unit --coordinate-origin checkpoint --difference-step .0001 --max-evaluations 8 --workers 28 --output outputs/stress-retrain
python -B compute/scripts/train_winhausen_neural.py initialize --initialize-from compute/parameters/joint_start.json --channels joint --stage-increment-weight 1 --samples 200 --max-increment .00003125 --workers 7 --output outputs/joint-retrain
python -B compute/scripts/train_winhausen_neural.py train --channels joint --block joint --optimizer least-squares --trust-solver lsmr --coordinate-scaling unit --coordinate-origin checkpoint --difference-step .0001 --max-evaluations 2 --workers 28 --output outputs/joint-retrain
```

All 46 material/evolution coordinates participate. The supplied final inputs
reproduce the manuscript histories directly. Reproduce the complete control
histories by using `joint_control.json` with the same fine/adaptive tracing
command and a separate output directory.
