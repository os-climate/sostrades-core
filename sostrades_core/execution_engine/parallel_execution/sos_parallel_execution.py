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
import logging

from gemseo.core.parallel_execution import DiscParallelExecution,\
    DiscParallelLinearization
import multiprocessing as mp

VALUE = 'value'
VAR_NAME = 'var_name'
TYPE_METADATA = "type_metadata"


class SoSDiscParallelExecution(DiscParallelExecution):
    """
    Execute disciplines in parallel
    """
    N_CPUS = mp.cpu_count()

    def __init__(self, worker_list, n_processes=N_CPUS, use_threading=False,
                 wait_time_between_fork=0):
        DiscParallelExecution.__init__(self, worker_list, n_processes=n_processes, use_threading=use_threading,
                                       wait_time_between_fork=wait_time_between_fork)
        self.logger = logging.getLogger(__name__)

    def _update_local_objects(self, ordered_outputs):
        """Update the local objects from parallel results.

        The ordered_outputs contains the stacked outputs of the function
        _run_task()

        :param ordered_outputs: the list of outputs, map of _run_task
            over inputs_list
        """
        for disc, output in zip(self.worker_list, ordered_outputs):

            # Update discipline local data
            local_data = output[0]
            #self.local_data.update(local_data)
            # Update values and metadata in DM
            # TODO: we should do a dm merge?
            # update values
            dm_data = output[1]
            update_dm_with_worker_results(dm_data, local_data, disc)

            # Update discipline status
            status_dict = output[2]
            disc.ee.load_disciplines_status_dict(status_dict)

    @staticmethod
    def _run_task(worker, input_loc):
        """Effectively performs the computation.

        To be overloaded by subclasses.

        :param worker: the worker pointes
        :param input_loc: input of the worker
        """
        if hasattr(worker, "execute"):
            worker.execute(input_loc)
            return get_data_from_worker(worker)
        if callable(worker):
            return worker(input_loc)  # , _, _
        raise TypeError("cannot handle worker")

    @staticmethod
    def _filter_ordered_outputs(ordered_outputs):
        """Filters the ordered_outputs.

        Eventually return a subset in the execute method.
        To be overloaded by subclasses.

        :param ordered_outputs: the list of outputs, map of _run_task
           over inputs_list
        """
        # Only keep the local_daya as outputs, dismiss local_data
        return [out[0] for out in ordered_outputs]

    def execute(self, input_data_list, exec_callback=None,
                task_submitted_callback=None):
        ''' execution overload to set all statuses as running
        '''
        for disc in self.worker_list:
            # rercursively updates status of disc and subdisc
            disc.update_status_running()

        self.logger.info(
            f"Parallel execution of {len(self.worker_list)} disciplines with {self.n_processes} processes")
#         for w in self.worker_list:
#             self.logger.info("\t " + w.get_disc_full_name())

        return DiscParallelExecution.execute(self, input_data_list, exec_callback,
                                             task_submitted_callback)


class SoSDiscParallelLinearization(DiscParallelLinearization):
    """Linearize disciplines in parallel."""
    N_CPUS = mp.cpu_count()

    def __init__(self, worker_list, n_processes=N_CPUS, use_threading=False,
                 wait_time_between_fork=0):
        DiscParallelLinearization.__init__(self, worker_list, n_processes=n_processes, use_threading=use_threading,
                                           wait_time_between_fork=wait_time_between_fork)
        self.logger = logging.getLogger(__name__)
        self.force_no_exec = False
        self.exec_before_linearize = True

    def configure_linearize_options(self, force_no_exec=False, exec_before_linearize=True):
        '''
        Configure options for the call to linearize in parallel for workers
        '''
        self.force_no_exec = force_no_exec
        self.exec_before_linearize = exec_before_linearize

    def _update_local_objects(self, ordered_outputs):
        """Update the local objects from parallel results.

        The ordered_outputs contains the stacked outputs of the function
        _run_task()

        :param ordered_outputs: the list of outputs, map of _run_task
            over inputs_list
        """
        for disc, output in zip(self.worker_list, ordered_outputs):
            # Update discipline jacobian
            disc.jac = output[0]

            local_data = output[1]

            dm_data = output[2]
            update_dm_with_worker_results(dm_data, local_data, disc)

    def _run_task_by_index(
        self, task_index  # type: int
    ):  # type: (...) -> Tuple[int, Any]
        """Run a task from an index of discipline and the input local data.

        The purpose is to be used by multiprocessing queues as a task.

        Args:
            task_index: The index of the task among `self.worker_list`.

        Returns:
            The task index and the output of its computation.
        """
        input_loc = self.input_data_list[task_index]
        if DiscParallelLinearization._is_worker(self.worker_list):
            worker = self.worker_list
        elif len(self.worker_list) > 1:
            worker = self.worker_list[task_index]
        else:
            worker = self.worker_list[0]

        # return the worker index to order the outputs properly
        output = self._run_task(worker, input_loc, force_no_exec=self.force_no_exec, exec_before_linearize=self.exec_before_linearize
                                )
        return task_index, output

    @staticmethod
    def _run_task(worker, input_loc, force_no_exec=False,
                  exec_before_linearize=True):
        """Effectively performs the computation.

        To be overloaded by subclasses

        :param worker: the worker pointes
        :param input_loc: input of the worker
        """
        jac = worker.linearize(input_loc, force_no_exec=force_no_exec,
                               exec_before_linearize=exec_before_linearize)
        local_data, dm_data, status_data = get_data_from_worker(worker)
        return jac, local_data, dm_data, status_data

    @staticmethod
    def _filter_ordered_outputs(ordered_outputs):
        """Filters the ordered_outputs.

        Eventually return a subset in the execute method.
        To be overloaded by subclasses.

        :param ordered_outputs: the list of outputs, map of _run_task
           over inputs_list
        """
        # Only keep the jacobians as outputs, dismiss local_data
        return [out[0] for out in ordered_outputs]

    def execute(self, input_data_list, exec_callback=None,
                task_submitted_callback=None):
        ''' execution overload to set all statuses as running
        '''

        self.logger.info(
            f"Parallel linearize of {len(self.worker_list)} disciplines with {self.n_processes} processes")
#         for w in self.worker_list:
#             self.logger.info("\t " + w.get_disc_full_name())

        return DiscParallelLinearization.execute(self, input_data_list, exec_callback,
                                                 task_submitted_callback)


def get_data_from_worker(worker):

    sub_disc = worker.get_sub_disciplines()
    all_discs = sub_disc + [worker]
    dm_data = worker.ee.dm.get_io_data_of_disciplines(all_discs)
    status_data = worker.ee.dm.build_disc_status_dict(all_discs)

    return dm_data.pop('local_data'), dm_data, status_data


def update_dm_with_worker_results(dm_data, local_data, disc):
    sub_disc = disc.get_sub_sos_disciplines()
    all_discs = sub_disc + [disc]
    dm_values = dm_data[VALUE]
    for d in all_discs:
        dm = d.ee.dm
        get_data = dm.get_data
        #- update data out
        data_out = d.get_data_out()
        out_f_keys = [d.get_var_full_name(
            k, data_out) for k in data_out.keys()]
        d_values = {get_data(k, VAR_NAME): dm_values[k] for k in out_f_keys}
        d.store_sos_outputs_values(d_values, update_dm=True)
        #- update data in
        data_in = d.get_data_in()
        in_f_keys = [d.get_var_full_name(
            k, data_in) for k in data_in.keys()]
        for full_k in in_f_keys:
            dm.set_data(full_k, VALUE,
                        dm_values[full_k], check_value=False)
        # TODO: won't work in case of var names updated by scatterDisciplines since
        # the link between data_i/o and DM is broken
        #d_values = {get_data(k, VAR_NAME): dm_values[k] for k in in_f_keys}
        #d._update_with_values(data_in, d_values)
        #- update GEMS i/o values
        loc_data = {k: v for k, v in local_data.items(
        ) if k in d.get_input_output_data_names()}
        d.local_data.update(loc_data)
        #disc.local_data.update(d.local_data)

    # update metadata
    dm_metadata = dm_data[TYPE_METADATA]
    for full_k, val in dm_metadata.items():
        disc.ee.dm.set_data(full_k, TYPE_METADATA,
                            val, check_value=False)
