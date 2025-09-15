'''
Copyright 2024 Capgemini
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

import inspect
from typing import Any, Callable, Dict, get_type_hints

from sostrades_core.execution_engine.sos_wrapp import SoSWrapp
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter

# Global namespace for auto disciplines
AUTO_DISCIPLINE_NAMESPACE = 'ns_global_autodiscipline'

"""
Decorator for automatically creating SoSTrades disciplines from simple functions.

This module provides:
- auto_sos_discipline: minimalist decorator to convert functions into SoSTrades disciplines
- All variables are automatically placed in the AUTO_DISCIPLINE_NAMESPACE namespace
- Supporting functions for parsing function signatures and type hints
"""


def auto_sos_discipline(func: Callable | None = None, *, outputs: Dict[str, Any] = None,
                        post_processing_callback: Callable | None = None,
                        namespace: str = AUTO_DISCIPLINE_NAMESPACE) -> type:
    """
    Decorator to automatically create a SoSTrades discipline from a simple function.

    All inputs are automatically inferred from the function signature and type hints.
    All variables are placed in the specified namespace (defaults to AUTO_DISCIPLINE_NAMESPACE).
    The function must return a dictionary and outputs must be specified.

    Args:
        func: The function to convert to a SoSTrades discipline
        outputs: Required dict specifying output names and Python types, e.g., {'fahrenheit': float, 'kelvin': float}
        post_processing_callback: Optional function that generates charts from input/output data
        namespace: Namespace for all variables, defaults to AUTO_DISCIPLINE_NAMESPACE

    Example:
        @auto_sos_discipline(outputs={'fahrenheit': float, 'kelvin': float})
        def temp_converter(celsius: float = 25.0) -> dict:
            return {'fahrenheit': celsius * 9/5 + 32, 'kelvin': celsius + 273.15}

        def custom_charts(data):
            return [chart1, chart2]

        @auto_sos_discipline(outputs={'fahrenheit': float, 'kelvin': float},
                           post_processing_callback=custom_charts)
        def temp_converter(celsius: float = 25.0) -> dict:
            return {'fahrenheit': celsius * 9/5 + 32, 'kelvin': celsius + 273.15}

    """
    def decorator(f: Callable) -> type:
        # Outputs parameter is now required
        if outputs is None:
            raise ValueError(f"The 'outputs' parameter is required for function '{f.__name__}'. "
                           f"Please specify the output names and types, e.g., "
                           f"@auto_sos_discipline(outputs={{'result': float}})")

        # Get function name and signature
        func_name = f.__name__
        signature = inspect.signature(f)
        type_hints = get_type_hints(f)

        # Parse inputs from function signature
        parsed_inputs = _parse_inputs_from_signature(signature, type_hints, namespace)

        # Parse outputs from the required outputs parameter
        parsed_outputs = {}
        for output_name, output_type in outputs.items():
            parsed_outputs[output_name] = {
                'type': _python_type_to_sos_type(output_type),
                'user_level': 1,
                'namespace': namespace
            }

        # Create default ontology data
        default_ontology = {
            'label': f'{f.__module__}.{func_name}',
            'type': 'Research',
            'source': 'SoSTrades Project',
            'validated': '',
            'validated_by': 'SoSTrades Project',
            'last_modification_date': '',
            'category': '',
            'definition': f.__doc__ or f'Automatically generated discipline from function {func_name}',
            'icon': 'fas fa-calculator fa-fw',
            'version': '',
        }

        # Create discipline class dynamically
        class AutoDiscipline(SoSWrapp):
            # Set class attributes
            _ontology_data = default_ontology
            _maturity = 'Research'
            DESC_IN = parsed_inputs
            DESC_OUT = parsed_outputs

            def __init__(self, sos_name, logger):
                super().__init__(sos_name, logger)
                self._user_function = f
                self._post_processing_callback = post_processing_callback

            def run(self):
                # Get inputs from SoSTrades
                input_values = {}
                for input_name in self.DESC_IN.keys():
                    input_values[input_name] = self.get_sosdisc_inputs(input_name)

                # Call user function with inputs
                result = self._user_function(**input_values)

                # Store outputs
                self.store_sos_outputs_values(result)

            def get_chart_filter_list(self):
                """Generate chart filters for post-processing"""
                if self._post_processing_callback is None:
                    return []

                # Create default chart filter
                chart_filters = []
                chart_list = [f'{func_name} charts']
                chart_filters.append(ChartFilter(
                    'Charts', chart_list, chart_list, 'graphs'))
                return chart_filters

            def get_post_processing_list(self, filters=None):
                """Generate post-processing charts using callback function"""
                if self._post_processing_callback is None:
                    return []

                # Check if charts are selected
                charts_list = []
                if filters is not None:
                    for chart_filter in filters:
                        if chart_filter.filter_key == 'graphs':
                            charts_list = chart_filter.selected_values
                else:
                    charts_list = [f'{func_name} charts']

                if f'{func_name} charts' not in charts_list:
                    return []

                # Get all inputs and outputs for post-processing
                all_data = {}
                for input_name in self.DESC_IN.keys():
                    all_data[input_name] = self.get_sosdisc_inputs(input_name)
                for output_name in self.DESC_OUT.keys():
                    all_data[output_name] = self.get_sosdisc_outputs(output_name)

                charts = self._post_processing_callback(all_data)
                return charts if isinstance(charts, list) else [charts]

        # Set the class name to match the function name
        AutoDiscipline.__name__ = func_name
        AutoDiscipline.__qualname__ = func_name

        return AutoDiscipline

    # Handle @auto_sos_discipline(outputs=...) - outputs parameter is required
    if func is None:
        return decorator
    else:
        # If called without parentheses, raise error since outputs is required
        raise ValueError("The 'outputs' parameter is required. Use @auto_sos_discipline(outputs={...}) "
                        "instead of @auto_sos_discipline")


def _parse_inputs_from_signature(signature: inspect.Signature, type_hints: Dict[str, Any], namespace: str) -> Dict[str, Dict[str, Any]]:
    """Parse input descriptors from function signature and type hints"""
    inputs = {}

    for param_name, param in signature.parameters.items():
        input_desc = {
            'user_level': 1,
            'namespace': namespace
        }

        # Get type from type hints or annotation
        param_type = type_hints.get(param_name, param.annotation)
        if param_type != inspect.Parameter.empty:
            input_desc['type'] = _python_type_to_sos_type(param_type)
        else:
            raise ValueError(f"No type hint found for parameter '{param_name}' in function '{signature}'. "
                           f"Please add a type hint, e.g., '{param_name}: float'")

        # Set default value if available
        if param.default != inspect.Parameter.empty:
            input_desc['default'] = param.default

        inputs[param_name] = input_desc

    return inputs


def _python_type_to_sos_type(python_type: Any) -> str:
    """Convert Python type to SoSTrades type string"""
    if python_type is int:
        return 'int'
    elif python_type is float:
        return 'float'
    elif python_type is str:
        return 'string'
    elif python_type is bool:
        return 'bool'
    elif python_type is list:
        return 'list'
    elif python_type is dict:
        return 'dict'
    elif hasattr(python_type, '__origin__'):
        # Handle generic types like list[float], dict[str, float], etc.
        origin = python_type.__origin__
        if origin is list:
            return 'list'
        elif origin is dict:
            return 'dict'
        else:
            # For other generic types, use the origin type name
            return getattr(origin, '__name__', str(origin))
    elif hasattr(python_type, '__name__'):
        name = python_type.__name__
        if name in ['DataFrame', 'dataframe']:
            return 'dataframe'
        elif name in ['ndarray', 'array']:
            return 'array'

    # Raise error for unsupported types
    raise ValueError(f"Unsupported Python type '{python_type}' for SoSTrades conversion. "
                     f"Supported types: int, float, str, bool, list, dict, DataFrame, ndarray, "
                     f"and generic types like list[T], dict[K, V]")
