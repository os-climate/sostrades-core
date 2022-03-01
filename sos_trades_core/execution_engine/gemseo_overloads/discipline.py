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
from timeit import default_timer as timer
from numpy import ndarray
from gemseo.core.discipline import MDODiscipline
from gemseo.utils.derivatives.derivatives_approx import DisciplineJacApprox


def execute(
    cls,
    input_data=None,  # type:Optional[Dict[str, Any]]
):  # type: (...) -> Dict[str, Any]
    """Execute the discipline.

    This method executes the discipline:

    * Adds the default inputs to the ``input_data``
      if some inputs are not defined in input_data
      but exist in :attr:`._default_inputs`.
    * Checks whether the last execution of the discipline was called
      with identical inputs, ie. cached in :attr:`.cache`;
      if so, directly returns ``self.cache.get_output_cache(inputs)``.
    * Caches the inputs.
    * Checks the input data against :attr:`.input_grammar`.
    * If :attr:`.data_processor` is not None, runs the preprocessor.
    * Updates the status to :attr:`.RUNNING`.
    * Calls the :meth:`._run` method, that shall be defined.
    * If :attr:`.data_processor` is not None, runs the postprocessor.
    * Checks the output data.
    * Caches the outputs.
    * Updates the status to :attr:`.DONE` or :attr:`.FAILED`.
    * Updates summed execution time.

    Args:
        input_data: The input data needed to execute the discipline
            according to the discipline input grammar.
            If None, use the :attr:`.default_inputs`.

    Returns:
        The discipline local data after execution.
    """
    # Load the default_inputs if the user did not provide all required data
    input_data = cls._filter_inputs(input_data)

    # Check if the cache already the contains outputs associated to these
    # inputs
    in_names = cls.get_input_data_names()

    # SoSTrades modif: cache capability removal
#     out_cached, out_jac = cls.cache.get_outputs(input_data, in_names)
#
#     if out_cached is not None:
#         __update_local_data_from_cache = getattr(
#             cls, "_MDODiscipline__update_local_data_from_cache")
#         __update_local_data_from_cache(input_data, out_cached, out_jac)
#         return cls.local_data
    # end of SoSTrades modif

    # Cache was not loaded, see self.linearize
    cls._cache_was_loaded = False

    # Save the state of the inputs
    __get_input_data_for_cache = getattr(
        cls, "_MDODiscipline__get_input_data_for_cache")
    cached_inputs = __get_input_data_for_cache(input_data, in_names)
    cls._check_status_before_run()

    cls.check_input_data(input_data)
    cls.local_data = {}
    cls.local_data.update(input_data)

    processor = cls.data_processor
    # If the data processor is set, pre-process the data before _run
    # See gemseo.core.data_processor module
    if processor is not None:
        cls.local_data = processor.pre_process_data(input_data)
    cls.status = cls.STATUS_RUNNING
    cls._is_linearized = False
    __increment_n_calls = getattr(cls, "_MDODiscipline__increment_n_calls")
    __increment_n_calls()
    t_0 = timer()
    try:
        # Effectively run the discipline, the _run method has to be
        # Defined by the subclasses
        cls._run()
    except Exception:
        cls.status = cls.STATUS_FAILED
        # Update the status but
        # raise the same exception
        raise
    __increment_exec_time = getattr(cls, "_MDODiscipline__increment_exec_time")
    __increment_exec_time(t_0)
    cls.status = cls.STATUS_DONE

    # If the data processor is set, post process the data after _run
    # See gemseo.core.data_processor module
    if processor is not None:
        cls.local_data = processor.post_process_data(cls.local_data)

    # Filter data that is neither outputs nor inputs
    cls._filter_local_data()

    cls.check_output_data()

    # Caches output data in case the discipline is called twice in a row
    # with the same inputs
    out_names = cls.get_output_data_names()
    cls.cache.cache_outputs(cached_inputs, in_names, cls.local_data, out_names)
    # Some disciplines are always linearized during execution, cache the
    # jac in this case
    if cls._is_linearized:
        cls.cache.cache_jacobian(cached_inputs, in_names, cls.jac)
    return cls.local_data


def check_jacobian(
    cls,
    input_data=None,  # type: Optional[Dict[str, ndarray]]
    derr_approx=MDODiscipline.FINITE_DIFFERENCES,  # type: str
    step=1e-7,  # type: float
    threshold=1e-8,  # type: float
    linearization_mode="auto",  # type: str
    inputs=None,  # type: Optional[Iterable[str]]
    outputs=None,  # type: Optional[Iterable[str]]
    parallel=False,  # type: bool
    n_processes=MDODiscipline.N_CPUS,  # type: int
    use_threading=False,  # type: bool
    wait_time_between_fork=0,  # type: float
    auto_set_step=False,  # type: bool
    plot_result=False,  # type: bool
    file_path="jacobian_errors.pdf",  # type: Union[str, Path]
    show=False,  # type: bool
    figsize_x=10,  # type: float
    figsize_y=10,  # type: float
    reference_jacobian_path=None,  # type: Optional[str, Path]
    save_reference_jacobian=False,  # type: bool
    indices=None,  # type: Optional[Iterable[int]]
):
    """Check if the analytical Jacobian is correct with respect to a reference one.

    If `reference_jacobian_path` is not `None`
    and `save_reference_jacobian` is `True`,
    compute the reference Jacobian with the approximation method
    and save it in `reference_jacobian_path`.

    If `reference_jacobian_path` is not `None`
    and `save_reference_jacobian` is `False`,
    do not compute the reference Jacobian
    but read it from `reference_jacobian_path`.

    If `reference_jacobian_path` is `None`,
    compute the reference Jacobian without saving it.

    Args:
        input_data: The input data needed to execute the discipline
            according to the discipline input grammar.
            If None, use the :attr:`.default_inputs`.
        derr_approx: The approximation method,
            either "complex_step" or "finite_differences".
        threshold: The acceptance threshold for the Jacobian error.
        linearization_mode: the mode of linearization: direct, adjoint
            or automated switch depending on dimensions
            of inputs and outputs (Default value = 'auto')
        inputs: The names of the inputs wrt which to differentiate the outputs.
        outputs: The names of the outputs to be differentiated.
        step: The differentiation step.
        parallel: Whether to differentiate the discipline in parallel.
        n_processes: The maximum number of processors on which to run.
        use_threading: Whether to use threads instead of processes
            to parallelize the execution;
            multiprocessing will copy (serialize) all the disciplines,
            while threading will share all the memory
            This is important to note
            if you want to execute the same discipline multiple times,
            you shall use multiprocessing.
        wait_time_between_fork: The time waited between two forks
            of the process / thread.
        auto_set_step: Whether to compute the optimal step
            for a forward first order finite differences gradient approximation.
        plot_result: Whether to plot the result of the validation
            (computed vs approximated Jacobians).
        file_path: The path to the output file if ``plot_result`` is ``True``.
        show: Whether to open the figure.
        figsize_x: The x-size of the figure in inches.
        figsize_y: The y-size of the figure in inches.
        reference_jacobian_path: The path of the reference Jacobian file.
        save_reference_jacobian: Whether to save the reference Jacobian.
        indices: The indices of the inputs and outputs
            for the different sub-Jacobian matrices,
            formatted as ``{variable_name: variable_components}``
            where ``variable_components`` can be either
            an integer, e.g. `2`
            a sequence of integers, e.g. `[0, 3]`,
            a slice, e.g. `slice(0,3)`,
            the ellipsis symbol (`...`)
            or `None`, which is the same as ellipsis.
            If a variable name is missing, consider all its components.
            If None,
            consider all the components of all the ``inputs`` and ``outputs``.

    Returns:
        Whether the analytical Jacobian is correct
        with respect to the reference one.
    """
    # Do not use self._jac_approx because we may want to check  complex
    # step approximation with the finite differences for instance
    approx = DisciplineJacApprox(
        cls,
        derr_approx,
        step,
        parallel,
        n_processes,
        use_threading,
        wait_time_between_fork,
    )
    if inputs is None:
        inputs = cls.get_input_data_names()
    if outputs is None:
        outputs = cls.get_output_data_names()

    if auto_set_step:
        approx.auto_set_step(outputs, inputs, print_errors=True)

    # Differentiate analytically
    cls.add_differentiated_inputs(inputs)
    cls.add_differentiated_outputs(outputs)
    cls.linearization_mode = linearization_mode
    cls.reset_statuses_for_run()
    # Linearize performs execute() if needed
    cls.linearize(input_data)

    def convert_jac(value):

        if isinstance(value, ndarray):
            return value
        else:
            return value.toarray()
    jac_arrays = {key_out: {key_in: convert_jac(value) for key_in, value in subdict.items()}
                  for key_out, subdict in cls.jac.items()}
    o_k = approx.check_jacobian(
        jac_arrays,
        outputs,
        inputs,
        cls,
        threshold,
        plot_result=plot_result,
        file_path=file_path,
        show=show,
        figsize_x=figsize_x,
        figsize_y=figsize_y,
        reference_jacobian_path=reference_jacobian_path,
        save_reference_jacobian=save_reference_jacobian,
        indices=indices,
    )
    return o_k


setattr(MDODiscipline, "execute", execute)
setattr(MDODiscipline, "check_jacobian", check_jacobian)
