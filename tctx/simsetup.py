from matplotlib import pyplot as plt

import pandas as pd
import numpy as np
from tqdm.auto import tqdm as pbar
import logging

from tctx.analysis import simbatch as sb
from tctx.networks import space_turtle as turtle
from tctx.util import sim
from tctx.analysis import simsampling


##################################################################################################################
# Original batch (F2 & F4)


def gen_instance_locs(folder, count):
    instance_idx = 0
    folder = sb.abs_path(folder)

    return [
        (str(folder / f'instance_{i:03d}.h5'), instance_idx)
        for i in range(count)
    ]


def new_nets(instance_locs):
    for i, (instance_path, instance_idx) in enumerate(pbar(instance_locs, desc='instantiate nets')):
        instance = turtle.get_network().instantiate()
        sb.CAT.remove_cats_cells(instance.cells)
        sb.CAT.remove_cats_conns(instance.connections)
        sim.save_instance(instance_path, instance, label=instance_idx, save_conns=True)


def sample_random_input_params(count, iwm_range, iws_range) -> pd.DataFrame:

    iwm = np.random.rand(count) * (iwm_range[1] - iwm_range[0]) + iwm_range[0]
    iws = np.random.rand(count) * (iws_range[1] - iws_range[0]) + iws_range[0]

    points = pd.DataFrame({
        'input_whitenoise_mean': iwm,
        'input_whitenoise_std': iws,
    })

    return points


def setup_all_sims(instance_locs, sim_count_per_net, iwm_range, iws_range) -> pd.DataFrame:

    all_entries = []

    for instance_path, instance_idx in pbar(instance_locs, desc='create sims'):

        instance = sim.load_instance(instance_path, instance_idx, load_conns=False)
        sb.CAT.add_cats_cells(instance.cells)

        params = instance.params.copy()
        params['targeted_gid'] = np.random.choice(instance.cells.index[instance.cells['ei_type'] == 'e'])
        params['instance_path'] = instance_path
        params['instance_idx'] = instance_idx
        params['instance_label'] = 'original'

        points = sample_random_input_params(sim_count_per_net, iwm_range, iws_range)

        for k, v in params.items():
            points[k] = v

        all_entries.append(points)

    return pd.concat(all_entries, ignore_index=True)


def specify_protocol(
    tstart=0,
    dt=.1,
    initial_delay=1000.,
    trial_count=100,
    duration_baseline=100.,
    duration_effect=300.,
    extra_after=1000.,
):
    trial_duration = duration_baseline + duration_effect
    end_of_trials = tstart + trial_duration * trial_count + initial_delay
    tend = tstart + end_of_trials + extra_after
    trial_times = tuple(np.arange(tstart + initial_delay, end_of_trials, trial_duration))

    dt_inv = 1. / dt
    assert float(int(dt_inv)) == dt_inv

    tstart_induction = np.min(trial_times) - duration_baseline
    tstop_induction = np.max(trial_times) + duration_effect

    protocol = {
        'trial_count': trial_count,
        'trial_length_ms': trial_duration,
        'duration_baseline': duration_baseline,
        'duration_effect': duration_effect,
        'tstart_induction': tstart_induction,
        'tstop_induction': tstop_induction,
        'tstart_pre': tstart,
        'tstop_pre': tstart_induction,
        'tstart_post': tstop_induction,
        'tstop_post': tend,
        'tstart': tstart,
        'dt': dt,
        'dt_inv': dt_inv,
        'tend': tend,
        'forced_times': trial_times
    }

    return protocol


def setup_original_batch(
        nets_folder,
        batch_desc,
        net_count=300,
        sim_count_per_net=20,
        iwm_range=(50, 110),
        iws_range=(0, 120),
):
    """

        setup_original_batch('nets_2021.12.01', 'original')

    """
    instance_folder = sb.abs_path(nets_folder)
    instance_folder.mkdir(exist_ok=True, parents=True)

    instance_locs = gen_instance_locs(instance_folder, net_count)

    new_nets(instance_locs)

    all_sims = setup_all_sims(instance_locs, sim_count_per_net, iwm_range, iws_range)

    sim_protocol = specify_protocol()

    for k, v in sim_protocol.items():
        all_sims[k] = [v] * len(all_sims)

    batch = simsampling.ProtocolSweep(all_sims)
    batch.describe_size()
    batch_path = batch.save(batch_desc)

    return batch_path


def plot_input_param_sampling(batch_results, iwm_range, iws_range, highlight=None, **kwargs):
    f, ax = plt.subplots(constrained_layout=True, figsize=(2, 1.5))

    ax.scatter(
        batch_results.reg['input_whitenoise_std'],
        batch_results.reg['input_whitenoise_mean'],
        clip_on=False,
        facecolor='xkcd:purple',
        **kwargs,
    )

    if highlight is not None:
        if isinstance(highlight, str):
            highlight = batch_results.reg[highlight]

            ax.scatter(
                batch_results.reg.loc[highlight, 'input_whitenoise_std'],
                batch_results.reg.loc[highlight, 'input_whitenoise_mean'],
                clip_on=False,
                facecolor='xkcd:light purple',
                edgecolor='k',
                linewidth=.5,
                zorder=1e6,
                **kwargs,
            )

    ax.set(
        xlim=iws_range,
        ylim=iwm_range,
        aspect='equal',
        ylabel=r'$\sigma_{in}$',
        xlabel=r'$\mu_{in}$',
    )


def randomly_pick(batch_results, count):
    count = min(len(batch_results), count)
    return np.random.choice(batch_results.reg.index, size=count, replace=False)


def randomly_tag(batch_results, count, col):
    which = randomly_pick(batch_results, count)

    if col in batch_results.reg.columns:
        logging.warning(f'Overwritting tag column: {col}')

    batch_results.reg[col] = False
    batch_results.reg.loc[which, col] = True


def save_batch(batch_results, working_filename):
    working_filename = sb.abs_path(working_filename)

    working_filename.parent.mkdir(parents=True, exist_ok=True)

    batch_results.save_registry(working_filename)
