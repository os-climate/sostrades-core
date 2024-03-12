# Wrap a model

## Definition
A SosWrap is a model wrapper for SoSTrades application

Here is the minimal working example of a SoSWrap :
```python
from sostrades_core.execution_engine.sos_wrapp import SoSWrapp
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import InstanciatedSeries, \
    TwoAxesInstanciatedChart
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter

class MyCustomWrap(SoSWrapp):
    # Ontology information
    _ontology_data = {
        'label': 'Label of the wrapp',
        'type': 'Research',
        'source': 'SoSTrades Project',
        'version': '',
    }


    # Description of inputs
    DESC_IN = {
        'x': {'type': 'float', 'default': 10, 'unit': 'year', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_one'},
        'a': {'type': 'float', 'unit': '-', 'namespace': 'ns_one'},
        'b': {'type': 'float', 'unit': '-',},
    }

    # Description of outputs
    DESC_OUT = {
        'y': {'type': 'float', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_one'}
    }

    # Method that runs the model
    def run(self):
        """
        Method that runs the model
        """
        # get input of discipline
        param_in = self.get_sosdisc_inputs()

        # performs the "computation"
        x = param_in['x']
        a = param_in['a']
        b = param_in['b']

        y = a * x + b

        output_values = {'y': y}

        # store data
        self.store_sos_outputs_values(output_values)
```

## Base class
The wrap should inherit from

```{eval-rst}
.. autoclass:: sostrades_core.execution_engine.sos_wrapp::SoSWrapp
```

## Ontology data

The ontology data specify all data regarding your SoSWrapp including : 
* `label` : Name of the wrapp on the ontology panel of the SoSTrades platform 
* `type` : Type of the model 'Research', 'Industrial' or 'Other'
* `source` : the person or project that has implemented the wrapp AND the model behind it
* `version` : A version of the model if necessary

## DESC_IN & DESC_OUT 

The DESC_IN and DESC_OUT dictionaries are the input and output variable descriptors. It gives information on variables in the wrapp used by the model. 

```python
DESC_IN = {
    'x': {'type': 'float', 'default': 10, 'unit': 'year', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_one'},
    'a': {'type': 'float', 'unit': '-', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_one'},
    'b': {'type': 'float', 'unit': '-',},
}
DESC_OUT = {
    'y': {'type': 'float', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_one'}
}
```

* `type` : mandatory could be : `'float'`, `'int'`, `'dict'`, `'dataframe'`, `'bool'`
* `subtype_descriptor` (or `dataframe_descriptor`) : if the variable is a dict/list (or dataframe), gives the types (or descriptor) of the sub-elements (or columns). See next sections
* `default` : if the variable has a default value. The default must be the same type as the type
* `unit` : (string) unity of the variable used for the ontology
* `user_level` : (optional) to filter the display in the GUI  (1=Standard by default, 2=Advanced, 3=Expert)
* `visibility`  : `'Shared'` if you need to specify a namespace for the variable or `'Local'` if the variable by default needs to be stored in the same namespace as the wrapp. If not specified the visibility is considered as `'Local'`.
* `namespace`  : must be identified by a string name, and its value must be defined within the process utilizing the wrapp. This feature allows for parameterizing the variable's location based on the specific process.
* `user_level`  : Specify the display level in the GUI: 1 for Standard view, 2 for Advanced, and 3 for Expert. If a variable is assigned an expert user level, it will only be visible in the expert view. This feature is useful for concealing complex variables that may be challenging to define. By default the display levvel is 1. 
* `range` : for float or int, range of the variable. the range will be checked by a data integrity method
* `possible_values` : for string, possible values list of the variable. the possible values will be checked by a data integrity method
* `optional` : A boolean flag that makes a variable optional to fill in the GUI
* `editable` : A boolean flag that makes a variable editable or not in the GUI. By default input and coupling variables are editable, outputs are not.
* `structuring` : A boolean flag that  that defines a structuring variable, indicating its impact on the configuration of the wrapp or other variables within the wrapp. For instance, it may be used for an assumption flag, and when activated, it creates new variables.



## Dataframe Descriptors
Here is an example dataframe descriptor. For each column you define a tuple which defines: 
-  first the type of the values in the column, 
-  second the range (for int or float) or possible values (for string), None if nothing is specified 
- third if the column is editable in the GUI or not.

```python
TransportChoiceData = {
    "var_name": "transport_choice",
    "type": "dataframe",
    "dataframe_descriptor" : {
        Years : ('int', YearsBoundaries, True),
        ProductName : ('string', None, True),
        TypeName : ('string', TransportPossibleValues, True),
        PercentageName : ('float', None, True),
    }
}
```

## Subtype descriptor for dicts
Here is an example of dict subtype descriptors. You can define an infinite depth for dictionaries and the type at the lower level will be checked.

```python
"dict_of_dict_in" : {"type": "dict", ProxyDiscipline.SUBTYPE: {"dict": {"dict": "float"}}, "user_level": 1}
"dict_of_dataframe_in" : {"type": "dict", ProxyDiscipline.SUBTYPE: {"dict": {"dataframe"}}, "user_level": 1}
```


## Run method

```python
# Method that runs the model
def run(self):
    """
    Method that runs the model
    """
    # get input of discipline
    param_in = self.get_sosdisc_inputs()

    # performs the "computation"
    x = param_in['x']
    a = param_in['a']
    b = param_in['b']

    y = a * x + b

    output_values = {'y': y}

    # store data
    self.store_sos_outputs_values(output_values)
```

* The function get_sosdisc_inputs(variable name) returns the value of the variable in the data manager. It can be used without arguments : return a dict with all keys and values of the DESC_IN
* The core of the model can be written here or loaded from an external model 
* Output values are stored in a dictionary {variable_name : value} with the value coming from the model
* The dictionary is sent to the data manager with the function store_sos_output_values(dict_values)


## Gradients computation method

### Context
In some situations, you may want to implement analytical gradients of some outputs of your model, with respect to given inputs.

Gradients are indeed required if, for example, you want to solve a Multidisciplinary Design Analysis (MDA) by using numerical methods like Newton-Raphson. Gradients are also involved by gradient-based optimization solvers (e.g., SLSQP, L-BFGS-B) to solve optimization problems.

Gradients can be computed automatically by finite differences (or complex step) by the core execution engine. However, this method can be costly in terms of number of calls to the discipline (cost linearly dependent to the number of inputs and outputs of the model).
The model developer can also implement its own [analytical gradient](#analytic-gradient-computation-method) formula in the model.

In the WITNESS framework for example, analytical gradients are involved at both optimization and MDA levels. It allows to reduce the execution time. This is why it is asked to contributors to update/implement the gradients corresponding to their contribution.

### Analytic gradient computation method

You need to implement the gradient in a method named `compute_sos_jacobian`, in the model wrap.

In this method, you can set the gradients of variables of type numerical like (1D) `array` as follows :

```python

def compute_sos_jacobian(self):
    """
    Analytic gradients computation
    """

    # retrieve the model input values
    param_in = self.get_sosdisc_inputs()

    # set the gradient values
    self.set_partial_derivative('y', 'x', atleast_2d(array(param_in['a'])))
    self.set_partial_derivative('y', 'a', atleast_2d(array(param_in['x'])))
    self.set_partial_derivative('y', 'b', atleast_2d(array([1])))
```

For gradients involving `dataframe`, `dict`, `float` types, you have to call the method `set_partial_derivative_for_other_types`.

For example, if you want to compute the gradient of a variable `y_2` (a dataframe with a column `value`) with respect to an array `z` :
```python
self.set_partial_derivative_for_other_types(('y_2', 'value'), ('z',), my_gradient_value)
```

Example of gradients can be found in the implementation of [these examples](https://github.com/os-climate/sostrades-core/blob/main/sostrades_core/sos_wrapping/test_discs/sellar_new_types.py).


