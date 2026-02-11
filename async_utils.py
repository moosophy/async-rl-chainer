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
    """
    Set optimizer states from shared arrays.
    For Chainer 7.8.1, optimizer states are stored in param.update_rule.state
    """
    assert isinstance(optimizer, chainer.Optimizer)
    assert hasattr(optimizer, 'target'), 'Optimizer.setup must be called first'
    
    model = optimizer.target
    
    # Map parameter names to params for quick lookup
    param_dict = {name: param for name, param in model.namedparams()}
    
    for param_name, shared_state in shared_arrays.items():
        if param_name not in param_dict:
            continue
            
        param = param_dict[param_name]
        
        # In Chainer 7.8.1, state is in param.update_rule.state
        if not hasattr(param, 'update_rule') or param.update_rule.state is None:
            continue
            
        for key, raw_array in shared_state.items():
            if key in param.update_rule.state:
                old_value = param.update_rule.state[key]
                param.update_rule.state[key] = np.frombuffer(
                    raw_array,
                    dtype=old_value.dtype
                ).reshape(old_value.shape)


def extract_params_as_shared_arrays(link):
    """
    Extract model parameters as shared arrays.
    Uses parameter names (strings) as keys for compatibility.
    """
    assert isinstance(link, chainer.Link)
    shared_arrays = {}
    for param_name, param in link.namedparams():
        shared_arrays[param_name] = mp.RawArray('f', param.data.ravel())
    return shared_arrays


def share_params_as_shared_arrays(link):
    shared_arrays = extract_params_as_shared_arrays(link)
    set_shared_params(link, shared_arrays)
    return shared_arrays


def share_states_as_shared_arrays(optimizer):
    shared_arrays = extract_states_as_shared_arrays(optimizer)
    set_shared_states(optimizer, shared_arrays)
    return shared_arrays


def extract_states_as_shared_arrays(optimizer):
    """
    Extract optimizer states as shared arrays.
    For Chainer 7.8.1, states are in param.update_rule.state
    """
    assert isinstance(optimizer, chainer.Optimizer)
    assert hasattr(optimizer, 'target'), 'Optimizer.setup must be called first'
    
    model = optimizer.target
    shared_arrays = {}
    
    for param_name, param in model.namedparams():
        # In Chainer 7.8.1, check if param has update_rule and state
        if not hasattr(param, 'update_rule'):
            continue
            
        if param.update_rule.state is None:
            # Initialize state if not already done
            if hasattr(param.update_rule, 'init_state'):
                param.update_rule.init_state(param)
            else:
                continue
        
        state = param.update_rule.state
        if state is None:
            continue
            
        shared_arrays[param_name] = {}
        
        for key, value in state.items():
            if isinstance(value, np.ndarray):
                shared_arrays[param_name][key] = mp.RawArray(
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
