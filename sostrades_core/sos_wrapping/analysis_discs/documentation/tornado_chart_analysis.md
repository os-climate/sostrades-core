The Tornado Chart analysis determines how different values of an independent variable affect a particular dependent variable (quantity of interest) under a given set of assumptions. The result can be visualized using a set of charts (tornado charts) that show the influence of each input variable on the quantities of interest, ranked by influence.



The Tornado chart analysis functionality in SoSTrades is accessed by creating a driver in a process, with sample generator and sampling method = "tornado chart analysis". It allows to measure the effect of input variations on outputs of the subprocess, and to automatically generate the Tornado charts.

A given percentage variation (+-X%) is considered for each input under study in the vicinity of the data point given by the subprocess original input. Each input is considered independently, in the fashion of a pattern search. Percentage variations are applied to input absolute values (same percentages for all inputs studied). 

At the selection of the 'tornado chart analysis" in the sample generator, a post processing namespace is added to the Eval namespace of the driver.

### Inputs configuration


- **evaluated inputs**: it is a default input of a sample generator that allows the user to select the inputs on which they want to perform the analysis. For now, the input type should be floats.

![1.1](evaluated_inputs.PNG)

- **gather outputs**: it is a default input of the driver evaluator that allows the user to select outputs as quantities of interest. Outputs can be floats, dict of floats, dataframe of floats.

- **variation list** in %: it is the list of percentages that will be applied to the inputs values to compute the variations.

### Outputs variables

-**scenario variations** in %: it is a non editable input that is computed at configuration time that gives for each scenario the percentages used.

-**samples df**: is in input in case of configuration at configuration time and in output in case of configuration at run time. It will be not editable except the names of the scenario. All the scenario will always be all selected.

-**Post processings**: a new namespace of post processing is added to show a tornado chart per selected outputs. The charts will show the variations for each type of outputs.

### Computation

The samples are computed like that:

-The values of the selected inputs are taken in the dm of the reference scenario

-The default values are in the first scenario, it will be the reference scenario.

-Then for each variation, a scenario is created for each values, the percentage is applied on only one variable per scenario

**Example**:

variation_list_in_% : [ -5.0, 5.0 ]

evaluated_inputs: [ a , b ] with values a=1.0, b=100.0

```
samples_df = {  'scenario_reference' : {a: 1.0, b: 100.0 },
                'scenario_1' : {a: 0.95, b: 100.0 },
                'scenario_2' : {a: 1.0, b: 95.0 },
                'scenario_3' : {a: 1.05, b: 100.0 },
                'scenario_4' : {a: 1.0, b: 105.0}}
```

And the corresponding scenario_variations:

```
scenario_variations  = { 'scenario_reference' : {a: 0, b: 0 },
                         'scenario_1' : {a: -5.0, b: 0},
                         'scenario_2' : {a: 0, b: -5.0 },
                         'scenario_3' : {a: 5.0, b: 0},
                         'scenario_4' : {a: 0, b: 5.0 }}
```

### Charts 


For now, only float outputs are handled in charts.

We make 1 graph for each output, with 1 line for each input :

![1.1](tornado_chart.PNG)
### Particular case : The output value of the reference scenario case is zero

When an output value of the reference scenario case is zero, it is not possible to compute a variation percentage with respect to the zero reference value (division by zero).

Then the string 'N/A' for not applicable is written in the output results. 
An empty chart is created for this output where the title defines that the computation si not possible with this reference value. 
