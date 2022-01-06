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
from gemseo.core.discipline import MDODiscipline

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
#         self.__update_local_data_from_cache(input_data, out_cached, out_jac)
#         return cls.local_data
    # end of SoSTrades modif

    # Cache was not loaded, see self.linearize
    cls._cache_was_loaded = False

    # Save the state of the inputs
    __get_input_data_for_cache = getattr(cls, "_MDODiscipline__get_input_data_for_cache")
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


setattr(MDODiscipline, "execute", execute)
