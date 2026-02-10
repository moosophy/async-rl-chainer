import multiprocessing as mp
import os
import random

import chainer
import numpy as np

import random_seed


def set_shared_params(a, b):
    """
    Args:
      a (chainer.Link): link whose params are to be replaced
      b (dict): dict that consists of (param_name, multiprocessing.Array)
    """
    assert isinstance(a, chainer.Link)
    for param_name, param in a.namedparams():
        if param_name in b:
            shared_param = b[param_name]
            param.data = np.frombuffer(
                shared_param, dtype=param.data.dtype).reshape(param.data.shape)


def set_shared_states(optimizer, shared_arrays):
    import numpy as np

    for idx, shared_state in shared_arrays.items():
        for key, raw_array in shared_state.items():
            old_param = optimizer._states[idx][key]
            optimizer._states[idx][key] = np.frombuffer(
                raw_array,
                dtype=old_param.dtype
            ).reshape(old_param.shape)



def extract_params_as_shared_arrays(link):
    assert isinstance(link, chainer.Link)
    shared_arrays = {}
    for param_name, param in link.namedparams():
        shared_arrays[id(param)] = mp.RawArray('f', param.data.ravel())
    return shared_arrays


def share_params_as_shared_arrays(link):
    shared_arrays = extract_params_as_shared_arrays(link)
    set_shared_params(link, shared_arrays)
    return shared_arrays


def share_states_as_shared_arrays(link):
    shared_arrays = extract_states_as_shared_arrays(link)
    set_shared_states(link, shared_arrays)
    return shared_arrays


#FUNCTION BELOW IS UPDATED FOR CHAINER 7.8.1
def extract_states_as_shared_arrays(optimizer):
    import multiprocessing as mp

    model = optimizer.target
    shared_arrays = {}

    for idx, param in enumerate(model.params()):
        state = param.update_rule.state
        if state is None:
            continue

        shared_arrays[idx] = {}

        for key, value in state.items():
            shared_arrays[idx][key] = mp.RawArray(
                'f',
                value.ravel()
            )

    return shared_arrays



def run_async(n_process, run_func):
    """Run experiments asynchronously.

    Args:
      n_process (int): number of processes
      run_func: function that will be run in parallel
    """

    processes = []

    def set_seed_and_run(process_idx, run_func):
        random_seed.set_random_seed(np.random.randint(0, 2 ** 32))
        run_func(process_idx)

    for process_idx in range(n_process):
        processes.append(mp.Process(target=set_seed_and_run, args=(
            process_idx, run_func)))

    for p in processes:
        p.start()

    for p in processes:
        p.join()
