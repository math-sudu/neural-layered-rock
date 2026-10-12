# Neural constitutive learning of strength and pore pressure

Reproduction materials for *Neural constitutive learning of strength and pore
pressure in saturated layered rock*, submitted to *Acta Geotechnica*.
Authors: Pengcheng Zhu and Tielin Chen. Corresponding author: Tielin Chen,
Key Laboratory of Urban Underground Engineering of Ministry of Education,
Beijing Jiaotong University, Beijing 100044, China; tlchen1@bjtu.edu.cn.
This repository contains the constitutive source, selected numerical inputs,
original-data retrieval script and commands for evaluating and training the model.

The model couples a monotone plane-storage
network and a pressure/history matrix-evolution network through a saturated
energy and an implicit constitutive update. Seven complete Opalinus Clay
specimens train the model; five specimens at other bedding orientations test
the selected stress-only and joint stress–pressure parameters.

## Environment

Tested with CPython 3.13.7, NumPy 2.4.4 and SciPy 1.16.2 on Windows x86-64.
The implementation uses double precision on the CPU. NumPy and SciPy are the
runtime dependencies; pytest runs the included implementation checks.
Run commands from the repository root containing this README and `compute/`.

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -r compute/requirements.txt
.venv/Scripts/python compute/scripts/fetch_winhausen_data.py
```

On POSIX, use `.venv/bin/python`. Below, `python` denotes that interpreter.
For several worker processes, set numerical-library threads to one to avoid
oversubscribing the CPU:

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
python -m pytest compute/tests -q
```

In a POSIX shell use `export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
MKL_NUM_THREADS=1` before the Python commands.

## Reproduce the reported histories

The two JSON inputs in `compute/parameters/` contain the full-precision
material constants and neural weights defining the reported constitutive laws.
Evaluation writes new CSV histories and JSON metrics under `outputs/` and
leaves those inputs unchanged. It does not fit the withheld observations.

```powershell
python compute/scripts/train_winhausen_neural.py evaluate --parameters compute/parameters/stress.json --channels stress --role test --samples 200 --max-increment .0000078125 --workers 5 --output outputs/stress
python compute/scripts/train_winhausen_neural.py evaluate --parameters compute/parameters/joint.json --channels joint --role test --samples 200 --max-increment .0000078125 --workers 5 --output outputs/joint
python compute/scripts/train_winhausen_neural.py evaluate --parameters compute/parameters/stress.json --channels stress --role train --samples 200 --max-increment .0000078125 --workers 7 --output outputs/stress
python compute/scripts/train_winhausen_neural.py evaluate --parameters compute/parameters/joint.json --channels joint --role train --samples 200 --max-increment .000015625 --workers 7 --output outputs/joint
```

Each test evaluation has 1,083 observation stations; each training evaluation
has 1,510. Substeps between stations resolve the constitutive history. Set
`--workers 1` for sequential execution. For the reported increment comparison,
double `--max-increment` and use a separate output directory. Training and test
files have different names and can share a model's output directory.

Expected withheld-history RMSEs in MPa, rounded to the manuscript precision:

| Specimen | Stress-only q | Stress-only pressure | Joint q | Joint pressure |
|---|---:|---:|---:|---:|
| OPA15 | 2.943 | 1.044 | 3.101 | 0.549 |
| OPA17 | 0.592 | 1.749 | 0.589 | 0.199 |
| OPA23 | 1.122 | 0.891 | 0.874 | 0.483 |
| OPA18 | 1.289 | 1.322 | 1.398 | 0.628 |
| OPA24 | 1.509 | 1.748 | 1.776 | 0.842 |

The JSON metrics include channel RMSEs, stress/pressure maxima and their strain
locations, pressure at the observed stress peak and at the endpoint, mechanism
work, damage increments and fluid/pressure identities. Predicted maxima include
every integration step; observed maxima use the entire retained source record.
CSV rows retain the original acquisition row, measured and imposed strains,
observed q/pressure, computed q/pressure and internal mechanism quantities.
The pressure-partition figure uses the OPA17 rows and the `pressure_stress_MPa`,
`pressure_matrix_MPa`, `matrix_volume` and `plane_slip` columns.

## Data and loading conventions

The data-retrieval command downloads twelve original files into
`compute/results/data/winhausen2023/raw/` from
[Winhausen et al. (2023)](https://doi.org/10.1029/2023JB027382) and their
[OSF repository](https://osf.io/5s2cb/), preserving the original bytes.
The adjacent `source.json` records each original download URL, specimen
identity, angle, strain rate, preparation pressure, fixed train/test split and
loading-segment exclusions. Cite the original experiment when reusing its data.

The model angle is the angle between bedding plane and loading axis, equal to
90 degrees minus the source angle. Raw strains are percentages and are converted
to dimensionless strain. Stress and pressure are in MPa; compression is positive.
The loader retains acquisition order and uses the initial absolute pressure as
the zero of excess pressure. The median of reconstructed cell pressure minus
initial pore pressure gives effective preparation confinement. The first
recorded q is applied as an elastic preload; subsequent loading is sealed.

Observation stations combine 200 uniform acquisition times, up to 16 rows
spaced uniformly in acquisition index in the first 0.2% axial strain, the
observed q peak and both endpoints. Repeated row selections are merged.
Pressure does not select them. A least-squares monotone strain estimate
handles recording jitter, while original rows and response targets remain.
OPA24 terminal unloading and OPA10's isolated terminal pressure collapse at
unchanged strain are excluded as specified in `source.json`. Original records
include those rows. The single recorded radial direction is not converted to
volume strain. Test strain rates vary with orientation.

The raw OPA10/OPA18 initial absolute pressure pairs differ from Table 3 of the
source article, while both still give about 10 MPa effective confinement.
The table's pressure maxima track absolute raw pressures rather than increments,
and its OPA15 stress-peak strain differs from the raw record. This reproduction
uses the original files and recomputes increments and peak strains. Full details
and exact terminal row boundaries are in `source.json`.

## Parameters and implementation

The 22 material coordinates are defined, bounded and decoded in
`compute/src/tfenn_invcalib/winhausen_training.py`. In particular,
`cross_fraction` maps to the manuscript's Poisson ratio as
`nu_pn = cross_fraction * sqrt((1 - nu_parallel) * E_parallel / (2 * E_normal))`.
`plane_compliance` is the dimensionless plane density D, and
`dilation_fraction` is beta_0/mu_0. Moduli in the JSON inputs are MPa.
Fixed constituent properties and the inactive cap are listed in `fixed_parameters`.

| Module | Constitutive role |
|---|---|
| `energetic_coupling.py` | Mandel geometry and common storage primitives |
| `poromechanical_coupling.py` | TI stiffness, saturated energy and conjugate forces |
| `coupled_friction_damage.py` | Implicit flow, contact, compaction and irreversible damage |
| `neural_storage.py` | Anchored monotone damage-to-plane-compliance network |
| `neural_evolution.py` | Effective-pressure/history resistance and volume-flow network |
| `winhausen_data.py` | Header-aware original-data loader and specimen split |
| `winhausen_training.py` | Loading history, sampling, material construction and loss |

At the zero-mean matrix-force corner, the implicit solve constrains the
friction multiplier to be nonnegative and the volume-flow fraction to lie
between zero and one. The original complementarity equations determine
acceptance. The included regression states exercise these transitions.

Storage weights are ordered as four raw output weights, four raw input slopes,
then four biases. Softplus makes the first two groups positive. Matrix weights
are the row-major 4-by-2 input matrix, four biases, then the row-major 2-by-4
output matrix. The density scale is `damage_peak_density`; the matrix pressure
scale is 10 MPa. No measured pressure is a network input.

## Training and selected continuation

The reported models were developed in the following sequence. All stages used
the seven training specimens; the five withheld histories were evaluated with
the selected parameters.

1. **Stress-only initialization and first fits.** The six stiffness coordinates
   were initialized from stress in the first 0--0.08% axial strain. Storage used
   the fixed weights in `MonotoneStorageNetwork.initialize`; the matrix network
   used the fixed hidden weights and zero output weights in
   `MatrixEvolutionNetwork.initialize`. An initial COBYQA fit adjusted all 54
   coordinates, followed by a material-only least-squares fit with both networks
   held fixed, then all-coordinate least-squares continuation. Every stage used
   stress targets alone.
2. **Stress-only continuation.** With the smooth directional evolution used in
   the paper, all-coordinate training continued on finer increments. A subsequent
   start set `damage_shape=0.85` to broaden the resistance profile, retaining the
   other learned values. This fit supplied the material and matrix network for
   `stress_start.json`; its storage weights were reset to the prescribed
   initialization before the final stress-only batch. This batch used an axial
   increment of 0.000015625 to resolve the OPA10 peak-stage sensitivity.
3. **Joint continuation.** The broader-resistance stress fit also initialized
   joint stress--pressure learning, retaining both networks. The material start
   used `matrix_cohesion=0.3` MPa, `matrix_friction=0.15`, `initial_density=0.6`,
   `E_normal=3000` MPa and `damage_resistance=0.035` MPa to promote earlier matrix
   activation. After all-coordinate fitting, the fourth matrix hidden unit's two
   input weights and bias were multiplied by 0.1 to broaden a resistance
   transition associated with OPA11 rehardening; its output weights were retained.
   All coordinates were then retrained. Saturated storage units motivated a final
   storage reset, retaining this learned material and matrix network, to form
   `joint_start.json`. The final joint batch again adjusted all 54 coordinates,
   at an axial increment of 0.00003125. The material and weight changes above
   set starting values; they did not hold those coordinates fixed during fitting.

Within each batch, the saved parameters minimize the training objective over
the evaluated candidates that completed all seven histories. The reported
stress fit was selected after finer-increment refitting resolved the OPA10
peak-stage sensitivity. The joint fit was selected for its lower refined
training objective, retained OPA11 postpeak softening and stable integration
under increment refinement. Withheld errors did not select either parameter set.

The reported comparison uses the two supplied selected parameter sets. Their
last training batches adjusted all 54 coordinates with complete-history finite
differences (step 1e-4 in optimization coordinates) and bounded reflective least
squares. Both stopped at the specified batch limit of 20 optimizer trials.
The stress-only batch made
736 candidate evaluations and rejected two unresolved histories; the joint
batch made 901 and had no unresolved histories. Their objective values were
4.60530851 (stress) and 14.07734711 (joint). The refined evaluations give
4.61516840 and 14.08120. The loss is a stage-weighted sum of squared MPa errors,
as defined by `residuals`.

The `stress.json` and `joint.json` inputs reproduce the reported forward
histories. The `stress_start.json` and `joint_start.json` inputs instead preserve
the material and network values immediately before the final continuation
batches. These starts include the preceding learning described above; they
replay the last batches rather than the entire development sequence. Replay
those batches in new experiment directories:

```powershell
python compute/scripts/train_winhausen_neural.py initialize --initialize-from compute/parameters/stress_start.json --channels stress --samples 200 --max-increment .000015625 --workers 7 --output outputs/stress-retrain
python compute/scripts/train_winhausen_neural.py train --channels stress --optimizer least-squares --max-evaluations 20 --workers 28 --output outputs/stress-retrain
python compute/scripts/train_winhausen_neural.py initialize --initialize-from compute/parameters/joint_start.json --channels joint --samples 200 --max-increment .00003125 --workers 7 --output outputs/joint-retrain
python compute/scripts/train_winhausen_neural.py train --channels joint --optimizer least-squares --max-evaluations 20 --workers 28 --output outputs/joint-retrain
```

For a new experiment from the implementation's canonical initialization, omit
`--initialize-from` and add `--matrix-evolution` to `initialize`; use a new
output directory. Such an experiment starts a different optimization path.
Material-only, storage-only and evolution-only updates are available through
`train --block material`, `storage` and `evolution`. The default `joint` block
adjusts all 54 coordinates. Numerical failures reject the candidate; accepted
paths retain their complete internal history. Withheld specimens enter only
the `evaluate --role test` command.
