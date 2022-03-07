'''
Copyright 2022 Airbus SAS

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''
# -*- coding: utf-8 -*-


import logging

import matplotlib.pyplot as plt
from numpy import array, concatenate, mean, isnan
from numpy.linalg import norm

from pandas import DataFrame
from gemseo.mda.mda import MDA
from copy import deepcopy

from gemseo.core.coupling_structure import MDOCouplingStructure
from gemseo.core.discipline import MDODiscipline
from sos_trades_core.execution_engine.sos_jacobian_assembly import SoSJacobianAssembly
from math import sqrt

LOGGER = logging.getLogger(__name__)


# Functions definition
def __init__(
    cls,
    disciplines,  # type: Sequence[MDODiscipline]
    max_mda_iter=10,  # type: int
    name=None,  # type: Optional[str]
    grammar_type=MDODiscipline.JSON_GRAMMAR_TYPE,  # type: str
    tolerance=1e-6,  # type: float
    linear_solver_tolerance=1e-12,  # type: float
    warm_start=False,  # type: bool
    use_lu_fact=False,  # type: bool
    coupling_structure=None,  # type: Optional[MDOCouplingStructure]
    log_convergence=False,  # type: bool
    linear_solver="LGMRES",  # type: str
    linear_solver_options=None,  # type: Mapping[str,Any]
    warm_start_threshold=-1
):  # type: (...) -> None
    # end of sostrades modif

    super(MDA, cls).__init__(name, grammar_type=grammar_type)
    cls.tolerance = tolerance
    cls.linear_solver = linear_solver
    cls.linear_solver_tolerance = linear_solver_tolerance
    cls.linear_solver_options = linear_solver_options or {}
    cls.max_mda_iter = max_mda_iter
    cls.disciplines = disciplines
    if coupling_structure is None:
        cls.coupling_structure = MDOCouplingStructure(disciplines)
    else:
        cls.coupling_structure = coupling_structure
    # SoSTrades modification
#     cls.assembly = JacobianAssembly(cls.coupling_structure)
    # end of SoSTrades modification
    cls.residual_history = []
    cls.reset_history_each_run = False
    cls.warm_start = warm_start

    # Don't erase coupling values before calling _compute_jacobian
    cls._linearize_on_last_state = True
    cls.norm0 = None
    cls.normed_residual = 1.0
    cls.strong_couplings = cls.coupling_structure.strong_couplings()
    cls.all_couplings = cls.coupling_structure.get_all_couplings()
    cls._input_couplings = []
    cls.matrix_type = SoSJacobianAssembly.SPARSE
    cls.use_lu_fact = use_lu_fact
    # By default dont use an approximate cache for linearization
    cls.lin_cache_tol_fact = 0.0

    cls._initialize_grammars()
    cls._check_consistency()
    check_linear_solver_options_func = getattr(
        cls, "_MDA__check_linear_solver_options")
    check_linear_solver_options_func()
    # cls._check_couplings_types()
    cls._log_convergence = log_convergence

    # SoSTrades modification
    if hasattr(cls, 'n_processes'):
        cls.assembly = SoSJacobianAssembly(
            cls.coupling_structure, cls.n_processes)
    else:
        cls.assembly = SoSJacobianAssembly(cls.coupling_structure)
    cls.cache_hist = None
    cls.warm_start_threshold = warm_start_threshold
    cls.linear_solver = linear_solver
    if cls.warm_start and warm_start_threshold != -1:
        LOGGER.info("Warm start residual threshold set at %s" %
                    warm_start_threshold)

    # Debug Variables
    cls.debug_mode_couplings = False
    # end of SoSTrades modification


def _current_input_couplings(cls):  # type: (...) -> ndarray
    """Return the current values of the input coupling variables."""
    input_couplings = list(iter(cls.get_outputs_by_name(cls._input_couplings)))
    if not input_couplings:
        return array([])
    concat_input_couplings = concatenate(input_couplings)

    if cls.debug_mode_couplings:
        print("IN CHECK of min/max couplings")
        plot_hist = False
        # plot history at first iteration
        if len(cls.residual_history) == 1:
            plot_hist = True
        cls._check_min_max_couplings(
            input_couplings, concat_input_couplings, plot_hist)

    return concat_input_couplings


def _check_min_max_couplings(cls, input_couplings, concat_input_couplings, plot_hist):
    '''Check of minimum and maximum coupling values 
    '''
    namespace_var_name_index = array([(var_name.rsplit('.', 1)[0], var_name.split('.')[-1], index) for var_name, value in zip(
        cls._input_couplings, input_couplings) for index in range(len(value))]).T
    min_max_couplings_df = DataFrame({'namespace': namespace_var_name_index[0],
                                      'var_name': namespace_var_name_index[1],
                                      'index': namespace_var_name_index[2],
                                      'abs_value': abs(concat_input_couplings)})
    sorted_couplings_df = min_max_couplings_df.sort_values(by=[
        'abs_value'])
    # add sorted coupling values to logger info
    LOGGER.info(
        f'Sorted coupling values: {sorted_couplings_df}')

    # plot history at first iteration
    if plot_hist:
        # compute mean, min, max for each coupling variable
        empty_index = [i for i, val in enumerate(
            input_couplings) if len(val) == 0]
        mean_couplings_df = DataFrame({'namespace': [var_name.split('.', 1)[-1].rsplit('.', 1)[0] for i, var_name in enumerate(cls._input_couplings) if i not in empty_index],
                                       'var_name': [var_name.split('.')[-1] for i, var_name in enumerate(cls._input_couplings) if i not in empty_index],
                                       'mean_value': [mean(val) for i, val in enumerate(input_couplings) if i not in empty_index],
                                       'min_value': [min(val) for i, val in enumerate(input_couplings) if i not in empty_index],
                                       'max_value': [max(val) for i, val in enumerate(input_couplings) if i not in empty_index]})
        # plot history for mean, min, max
        mean_couplings_df.hist(bins=20)
        plt.suptitle(
            'Coupling variables distribution at first iteration of the MDA')
        plt.show()

# fixed point methods


def _compute_residual(
    cls,
    current_couplings,  # type: ndarray
    new_couplings,  # type: ndarray
    current_iter,  # type: int
    first=False,  # type: bool
    store_it=True,  # type: bool
    log_normed_residual=False,  # type: bool
):  # type: (...) -> ndarray
    """Compute the residual on the inputs of the MDA.

    Args:
        current_couplings: The values of the couplings before the execution.
        new_couplings: The values of the couplings after the execution.
        current_iter: The current iteration of the fixed-point method.
        first: Whether it is the first residual of the fixed-point method.
        store_it: Whether to store the normed residual.
        log_normed_residual: Whether to log the normed residual.

    Returns:
        The normed residual.
    """
    if first and cls.reset_history_each_run:
        cls.residual_history = []

    normed_residual = norm(
        (current_couplings - new_couplings).real) / sqrt(current_couplings.size)
    if cls.norm0 is None:
        cls.norm0 = normed_residual
        if hasattr(cls, 'epsilon0'):
            if cls.epsilon0 is not None:
                cls.norm0 = cls.epsilon0
        else:
            LOGGER.warning(
                'epslion0 attribute is not set in case of MDF formulation')
    if cls.norm0 == 0:
        cls.norm0 = 1
    cls.normed_residual = normed_residual / cls.norm0

    # deactivated by default in SoSTrades core
    if log_normed_residual:
        LOGGER.info(
            "%s running... Normed residual = %s (iter. %s)",
            cls.name,
            "{:.2e}".format(cls.normed_residual),
            current_iter,
        )
    # added by SoSTrades
    res_norm = '{:e}'.format(cls.normed_residual)
    LOGGER.info(f'\t{current_iter}\t{res_norm}')
    # end of SoSTrades

    if store_it:
        cls.residual_history.append((cls.normed_residual, current_iter))

    max_residual_limit = 1e30
    if cls.normed_residual > max_residual_limit or isnan(cls.normed_residual):
        raise Exception(
            f'The residual on the inputs of the MDA is higher than {max_residual_limit}, try to understand the divergence or use another more robust MDA method')

    return cls.normed_residual


def store_state_for_warm_start(cls):
    ''' stores a state of the local_data if the residuals are lower than the threshold
    '''
    # if the current residual is lower than the residual threshold
    if cls.normed_residual <= cls.warm_start_threshold:
        cached_output = deepcopy(cls.local_data)
        # if there is no cache already stored
        if cls.cache_hist is None:
            cls.cache_hist = cls.normed_residual, cached_output
        # if already stored, check if the current residual is closer than
        # the one already stored
        else:
            current_delta = abs(cls.normed_residual -
                                cls.warm_start_threshold)
            stored_delta = abs(
                cls.cache_hist[0] - cls.warm_start_threshold)
            if current_delta <= stored_delta:
                cls.cache_hist = cls.normed_residual, cached_output


# Set functions to the MDA Class
#setattr(MDA, "__init__", __init__)
#setattr(MDA, "_current_input_couplings", _current_input_couplings)
#setattr(MDA, "_check_min_max_couplings", _check_min_max_couplings)
#setattr(MDA, "_compute_residual", _compute_residual)
#setattr(MDA, "store_state_for_warm_start", store_state_for_warm_start)
