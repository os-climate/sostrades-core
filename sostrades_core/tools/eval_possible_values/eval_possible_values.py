from sostrades_core.execution_engine.proxy_coupling import ProxyCoupling

EVAL_INPUT_TYPE = ['float', 'array', 'int', 'string']
MULTIPLIER_PARTICULE = '__MULTIPLIER__'
NUMERICAL_VAR_LIST = list(
    ProxyCoupling.DESC_IN.keys())


def find_possible_values(disc, prefix_name_to_delete,
                         io_type_in=True, io_type_out=True, strip_first_ns=False):
    '''
        Run through all disciplines and sublevels
        to find possible values for eval_inputs and eval_outputs
    '''
    # TODO: does this involve avoidable, recursive back and forths during  configuration ? (<-> config. graph)
    possible_in_values, possible_out_values = fill_possible_values(
        disc, prefix_name_to_delete, io_type_in=io_type_in, io_type_out=io_type_out)
    if hasattr(disc, 'scenarios'):
        sub_disciplines = disc.scenarios
    else:
        sub_disciplines = disc.proxy_disciplines

    for sub_disc in sub_disciplines:
        sub_in_values, sub_out_values = fill_possible_values(
            sub_disc, prefix_name_to_delete, io_type_in=io_type_in, io_type_out=io_type_out)
        possible_in_values.update(sub_in_values)
        possible_out_values.update(sub_out_values)
        sub_in_values, sub_out_values = find_possible_values(
            sub_disc, prefix_name_to_delete,
            io_type_in=io_type_in, io_type_out=io_type_out)
        possible_in_values.update(sub_in_values)
        possible_out_values.update(sub_out_values)

    # strip the scenario name to have just one entry for repeated variables in scenario instances
    if strip_first_ns:
        return [_var.split('.', 1)[-1] for _var in possible_in_values], [_var.split('.', 1)[-1] for _var in
                                                                         possible_out_values]
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
        # a possible input value must :
        #           - be a ['float', 'array', 'int', 'string']
        #           - be an input (not a coupling variable)
        #           - be editable
        #           - not be a numerical
        #           - not be structuring
        #           - not be a multiplier
        if is_in_type and key not in NUMERICAL_VAR_LIST and not is_structuring and is_editable and is_input_type and not is_a_multiplier:
            # we remove the disc_full_name name from the variable full  name for a
            # sake of simplicity
            poss_in_values_full.add(
                full_id.split(f'{prefix_name_to_delete}.', 1)[1])

    return poss_in_values_full


def fill_possible_output_values(disc, poss_out_values_full, prefix_name_to_delete):
    disc_out = disc.get_data_out()
    for data_out_key in disc_out.keys():
        # Caution ! This won't work for variables with points in name
        # as for ac_model
        full_id = disc.get_var_full_name(
            data_out_key, disc_out)
        if data_out_key != 'residuals_history':
            # we anonymize wrt. driver evaluator node namespace
            poss_out_values_full.add(
                full_id.split(f'{prefix_name_to_delete}.', 1)[1])

    return poss_out_values_full
