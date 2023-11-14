'''
Copyright 2023 Capgemini

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
from sostrades_core.execution_engine.proxy_coupling import ProxyCoupling

EVAL_INPUT_TYPE = ['float', 'array', 'int', 'string']
MULTIPLIER_PARTICULE = '__MULTIPLIER__'
NUMERICAL_VAR_LIST = ProxyCoupling.NUMERICAL_VAR_LIST


def find_possible_input_values(disc, prefix_name_to_delete=None, strip_first_ns=False):
    '''
    This method will find all possible input value under disc and all subdisciplines recursively
    Args:
        disc: disc where to find possible input values
        prefix_name_to_delete: prefix to delete before all input values name if None then prefix is disc.get_disc_full_name()
        strip_first_ns: If True will strip the first prefix by splitting with the point delimiter

    Returns:
        A set of possible input_values
    '''
    return find_possible_values(disc, prefix_name_to_delete=prefix_name_to_delete,
                                io_type_in=True, io_type_out=False, strip_first_ns=strip_first_ns)[0]


def find_possible_output_values(disc, prefix_name_to_delete=None, strip_first_ns=False):
    '''
    This method will find all possible output values under disc and all subdisciplines recursively
    Args:
        disc: disc where to find possible output values
        prefix_name_to_delete: prefix to delete before all output values name if None then prefix is disc.get_disc_full_name()
        strip_first_ns: If True will strip the first prefix by splitting with the point delimiter

    Returns:
        A set of possible output_values
    '''
    return find_possible_values(disc, prefix_name_to_delete=prefix_name_to_delete,
                                io_type_in=False, io_type_out=True, strip_first_ns=strip_first_ns)[1]


def find_possible_values(disc, prefix_name_to_delete=None, io_type_in=True, io_type_out=True, strip_first_ns=False):
    '''
        his method will find all possible output and inputs values under disc and all subdisciplines recursively
        Args:
        disc: disc where to find possible output values
        prefix_name_to_delete: prefix to delete before all output values name if None then prefix is disc.get_disc_full_name()
        strip_first_ns: If True will strip the first prefix by splitting with the point delimiter

    Returns:
        A set of possible input_values
        A set of possible output_values
    '''
    # if no prefix_name to delete has been filled we use the full_name of the disc
    if prefix_name_to_delete is None:
        prefix_name_to_delete = disc.get_disc_full_name()

    possible_in_values, possible_out_values = set(), set()
    # fill possiblee values set for the high level disc
    if disc.get_disc_full_name() != prefix_name_to_delete:
        possible_in_values, possible_out_values = fill_possible_values(
            disc, prefix_name_to_delete, io_type_in=io_type_in, io_type_out=io_type_out)

    # find sub_disciplines if it's a driver then subdisciplines are stored in scenarios (proxy in run with flatten subprocess)
    if hasattr(disc, 'scenarios'):
        sub_disciplines = disc.scenarios
    else:
        sub_disciplines = disc.proxy_disciplines

    # loop over all subdisciplines to find possible i/O values
    for sub_disc in sub_disciplines:
        sub_in_values, sub_out_values = fill_possible_values(
            sub_disc, prefix_name_to_delete, io_type_in=io_type_in, io_type_out=io_type_out)
        possible_in_values.update(sub_in_values)
        possible_out_values.update(sub_out_values)
        # Recursively if there is multiple levels
        sub_in_values, sub_out_values = find_possible_values(
            sub_disc, prefix_name_to_delete, io_type_in=io_type_in, io_type_out=io_type_out)
        possible_in_values.update(sub_in_values)
        possible_out_values.update(sub_out_values)

    # strip the scenario name to have just one entry for repeated variables in scenario instances
    if strip_first_ns:
        return {_var.split('.', 1)[-1] for _var in possible_in_values}, {_var.split('.', 1)[-1] for _var in
                                                                         possible_out_values}
    else:
        return possible_in_values, possible_out_values


def fill_possible_values(disc, prefix_name_to_delete, io_type_in=False, io_type_out=True):
    '''
        Fill possible values lists for eval inputs and outputs
        an input variable must be a float coming from a data_in of a discipline in all the process
        and not a default variable
        an output variable must be any data from a data_out discipline
    '''
    poss_in_values_full = set()
    poss_out_values_full = set()
    if io_type_in:  # TODO: edit this code if adding multi-instance eval_inputs in order to take structuring vars
        poss_in_values_full = fill_possible_input_values(disc, poss_in_values_full, prefix_name_to_delete)

    if io_type_out:
        poss_out_values_full = fill_possible_output_values(disc, poss_out_values_full, prefix_name_to_delete)

    return poss_in_values_full, poss_out_values_full


def fill_possible_input_values(disc, poss_in_values_full, prefix_name_to_delete):
    '''

    Args:
        disc: discipline where to find input values
        poss_in_values_full: list where to store input values
        prefix_name_to_delete: prefix_name_to_delete to delete from the name of the input value

    Returns:
        Set of possible input values
    '''
    disc_in = disc.get_data_in()
    for key, data_dict in disc_in.items():
        is_input_type = data_dict[ProxyCoupling.TYPE] in EVAL_INPUT_TYPE
        is_structuring = data_dict.get(
            ProxyCoupling.STRUCTURING, False)
        full_id = disc.get_var_full_name(
            key, disc_in)
        is_in_type = disc.dm.data_dict[disc.dm.data_id_map[full_id]
                     ]['io_type'] == 'in'
        is_editable = data_dict['editable']
        is_a_multiplier = MULTIPLIER_PARTICULE in key
        # is_numerical = data_dict.get(
        #             ProxyCoupling.NUMERICAL, False)

        # a possible input value must :
        #           - be a ['float', 'array', 'int', 'string']
        #           - be an input (not a coupling variable)
        #           - be editable
        #           - not be a numerical
        #           - not be structuring
        #           - not be a multiplier

        # NB: using ProxyCoupling.NUMERICAL_VAR_LIST implies subprocess driver & optim numerical input are not forbidden
        if is_in_type and key not in NUMERICAL_VAR_LIST and not is_structuring and is_editable and is_input_type and not is_a_multiplier:
            # we remove the disc_full_name name from the variable full  name for a
            # sake of simplicity

            poss_in_values_full.add(full_id.removeprefix(f'{prefix_name_to_delete}.'))

    return poss_in_values_full


def fill_possible_output_values(disc, poss_out_values_full, prefix_name_to_delete):
    '''

    Args:
        disc: discipline where to find output values
        poss_out_values_full: list where to store output values
        prefix_name_to_delete: prefix_name_to_delete to delete from the name of the output value

    Returns:
        Set of possible output values
    '''
    disc_out = disc.get_data_out()
    for data_out_key in disc_out.keys():
        # Caution ! This won't work for variables with points in name
        # as for ac_model
        full_id = disc.get_var_full_name(
            data_out_key, disc_out)
        if data_out_key != 'residuals_history':
            # we anonymize wrt. driver evaluator node namespace
            poss_out_values_full.add(
                full_id.removeprefix(f'{prefix_name_to_delete}.'))

    return poss_out_values_full
