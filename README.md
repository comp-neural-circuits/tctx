# TCTX

tctx is a Python library containing the code for the manuscript:

**Sequential propagation and routing of activity in a cortical network**

Juan Luis Riquelme, Mike Hemberger, Gilles Laurent, Julijana Gjorgjieva

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install tctx.

```bash
pip install tctx
```

## Usage

Simulations must be set up, run, post-processed.

### Simulation setup

Setup will instantiate networks and create a registry of simulations to be run, specifying all parameters. Networks and simulation batches are saved as HDF5 files. All setup code can be accessed via `tctx/simsetup.oy` following the steps in `tctx/notebooks/sims.ipynb`.

### Simulation run

Simulations are run with Nest simulator using the script `tctx/seqexp.py`. For example: 

```bash
nohup python tctx/experiments/seqexp.py registry.h5 -l info --redirect &> nohup_$HOSTNAME.log &
```

### Simulation results post-processing 

Post-processing extracts properties from the simulation results such as detecting followers from trigger neurons. Post-processing results are saved as HDF5 next to the simulation batch. All post-processing code can be accessed via `tctx/postpro.py` and following the steps in `tctx/notebooks/sims.ipynb`.
