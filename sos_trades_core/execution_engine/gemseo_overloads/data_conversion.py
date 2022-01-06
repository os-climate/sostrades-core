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
from gemseo.utils.data_conversion import DataConversion
from numpy import ndarray, array
from copy import deepcopy

@staticmethod
def update_dict_from_array(
    reference_input_data,  # type: Mapping[str,ndarray]
    data_names,  # type: Iterable[str]
    values_array,  # type: ndarray
):  # type: (...) -> Dict[str,ndarray]
    """Update a data mapping from data array and names.

    The order of the data in the array follows the order of the data names.

    Args:
        reference_input_data: The reference data to be updated.
        data_names: The names for which to update the data.
        values_array: The data with which to update the reference one.

    Returns:
        The updated data mapping.

    Raises:
        TypeError: If the data with which to update the reference one
            is not a NumPy array.
        ValueError:
            * If a name for which to update the data is missing
              from the reference data.
            * If the size of the data with which to update the reference one
              is inconsistent with the reference data.
    """
    if not isinstance(values_array, ndarray):
        raise TypeError(
            "Values array must be a numpy.ndarray, "
            "got instead: {}.".format(type(values_array))
        )

    data = dict(deepcopy(reference_input_data))

    if not data_names:
        return data

    i_min = i_max = 0
    for data_name in data_names:
        data_value = reference_input_data.get(data_name)
        if data_value is None:
            raise ValueError(
                "Reference data has no item named: {}.".format(data_name)
            )
        # SoSTrades fix
        if isinstance(data_value, list):
            data_value = array(data_value)
        #
        i_max = i_min + data_value.size
        if len(values_array) < i_max:
            raise ValueError(
                "Inconsistent input array size of values array {} "
                "with reference data shape {} "
                "for data named: {}.".format(
                    values_array, data_value.shape, data_name
                )
            )
        data[data_name] = values_array[i_min:i_max].reshape(data_value.shape)
        #- SoSTrades modif
        # we do NOT force type because of complex step. 
        # It allows to avoid to switch all the default inputs as complex
#             data[data_name] = data[data_name].astype(data_value.dtype)
        # end of SoSTrades modif
        i_min = i_max

    if i_max != values_array.size:
        raise ValueError(
            "Inconsistent data shapes:\n"
            "could not use the whole data array of shape {} "
            "(only reached max index = {}),\n"
            "while updating data dictionary keys {}\n"
            " of shapes : {}.".format(
                values_array.shape,
                i_max,
                data_names,
                [
                    (data_name, reference_input_data[data_name].shape)
                    for data_name in data_names
                ],
            )
        )

    return data


# Set functions to the MDA Class
setattr(DataConversion, "update_dict_from_array", update_dict_from_array)
