# Documentation of the Build Very Simple Multi-Scenario driver

## Short description

The build vs_MS discipline driver is a specialisation of the vs_MS driver. 
It extends the vs_MS discipline driver by providing the capability to select in the GUI the used sub_process of the driver and also the associated scenario_map. 

The vs_MS capability is dedicated to create as many scenarios as the user needs to perform on its nested sub_process.
It creates as many instances of the sub_process as selected scenarios. 
Each instance will have its inputs and its outcome outputs (after run).


## The several usage steps are detailed below.

![Empty vsMS](./sos_vs_MS_png/vs_MS_17.PNG)

###  Step 1: 
- 1.1 Select your sub_process_inputs: 
	- proces_repository and process_name (obligatory)
	- select a usecase to import data (optional)
	
![Empty vsMS](./sos_vs_MS_png/vs_MS_18.PNG)

![Empty vsMS](./sos_vs_MS_png/vs_MS_19.PNG)

You can also at this step select the data usecase for the reference scenario (or can do it later as seen in step 4)
- 1.2 Select your scenario_map: (obligatory)
	- input_name: it will be the name of "the dynamic input selector" to select the list of names of your scenario nodes
	- ns_to_update: it is the list of namespace used in the sub_process. If the step 1.1 has been done and configured then the full list of namespaces is provided after configuration, it will generate the dynamic input selector

![Empty vsMS](./sos_vs_MS_png/vs_MS_20.PNG)

![Empty vsMS](./sos_vs_MS_png/vs_MS_21.PNG)



Remark: you can do 
- (1.1 and 1.2) and configure
- 1.1 configure and then 1.2 configure
- 1.2 configure and then 1.1 configure

You finally are in the following state:
![Empty vsMS](./sos_vs_MS_png/vs_MS_22.PNG)

Remark: the instances of the subprocess will appear in the tree, only when a first name of scenario (or severals) is provided in "the dynamic input selector"

Remark: the subprocess will be build in memory only when the step 1.1 and 1.2 have been achieved and configured

###  Step 2: provide your list of name of your scenarios in the "the dynamic input selector"

![Empty vsMS](./sos_vs_MS_png/vs_MS_23.PNG)

![Empty vsMS](./sos_vs_MS_png/vs_MS_24.PNG)

![Empty vsMS](./sos_vs_MS_png/vs_MS_25.PNG)

Warning: you can created a "reference" scenario if you want to use the import of usecase capability

###  Step 3: provide data in your instantiated disciplines

###  Step 4: provide data in the reference scenario node by importing a usecase. 
![Empty vsMS](./sos_vs_MS_png/vs_MS_26.PNG)

Warning: you need to have first created a "reference" scenario by adding reference in the "the dynamic input selector"

Warning: this capability only works with a ns_to_update filled with all namespaces

###  Step 5 (optional): you can update the data of your reference usecase by importing a new usecase 

###  Step 6 (optional): you can decide to provide a new subprocess and associated scenario map and rebuild a new process

## Example of models to be nested

### csv files for sub_process_inputs and scenario_map_csv
In the "empty driver process" folder, csv files are provided for a list of nested models:
- sub_process_inputs_csv folder:

![sub_process_inputs](./sos_vs_MS_png/vs_MS_14.PNG)
- scenario_map_csv folder:

![scenario_map_csv](./sos_vs_MS_png/vs_MS_15.PNG)

### Models in usecases
In the "empty driver process" folder, complete usecases are provided for a list of nested models:

![Study creation](./sos_vs_MS_png/vs_MS_16.PNG)


## Current small restrictions
- import load case: it works only when a scenario of name 'reference' was already provided and the associated analysis already instantiated
- import load case: it works only when all the namespaces are selected in the ns_to_update input

## Future evolutions
- The scenario_map should be a dynamic entry that is created only when sub_process has been created [to be confirmed]
- In fact all the scatter map should be a (or several) Desc_in of the vsMS driver and SosScatterDiscipline should relies on dynamic 
   Desc_in entries (maybe expert depending on parameters) rather than on a stored scenario map [to be confirmed]
- Remove this step of scenario map in case of multiscenario with the same architecture for all scenarios 
(and so a reference scenario provided) [to be confirmed]
- Refactory of the very Simple Multiscenarios capability [to be confirmed]
- Reintroduce the optional parameters of the scenario map [to be confirmed]
- Allow to use the mode autogather = Yes [to be confirmed]
- Remove the current small restrictions [to be confirmed]
- Improve the maintainability of the test file
- Update the code with the new the modifications of doe_eval introduced in core by the web/Api




## Structure of Desc_in/Desc_out

        |_ DESC_IN
            |_ SUB_PROCESS_INPUTS (structuring)
            |_ SCENARIO_MAP (structuring)            
               |_ SCENARIO_MAP[INPUT_NAME] (namespace: INPUT_NS if INPUT_NS in SCENARIO_MAP keys / if not then  local,
                                                      structuring, dynamic: valid SCENARIO_MAP)
        |_ DESC_OUT
		
##     Description of DESC parameters
        |_ DESC_IN
           |_ SUB_PROCESS_INPUTS:  All inputs for driver builder in the form of a dictionary of four keys
                                   PROCESS_REPOSITORY:   folder root of the sub processes to be nested inside the driver.
                                                         If 'None' then it uses the sos_processes python for driver creation.
                                   PROCESS_NAME:         selected process name (in repository) to be nested inside the driver.
                                                         If 'None' then it uses the sos_processes python for driver creation.
                                   USECASE_NAME:         either empty or an available usecase of the sub_process
                                   USECASE_DATA:         anonymized dictionary of usecase inputs to be nested in context
                                                         it is a temporary input: it will be put to None as soon as                                                                        
                                                         its content is 'loaded' in the dm. We will have it has editable                                                                             
                                   It is in dict type (specific 'proc_builder_modale' type to have a specific GUI widget) 
           |_ SCENARIO_MAP:        All inputs for driver builder in the form of a dictionary of four keys
                                   INPUT_NAME:           name of the variable to scatter
                                   INPUT_NS:             Optional key: namespace of the variable to scatter if the INPUT_NS key is this scenario map.
                                                         If the key is not here then it is local to the driver.
                                   OUTPUT_NAME:          name of the variable to overwrite
                                   SCATTER_NS:           Optional key: Internal namespace associated to the scatter discipline
                                                         it is a temporary input: its value is put to None as soon as scenario disciplines are instantiated
                                   GATHER_NS:            Optional key: namespace of the gather discipline associated to the scatter discipline 
                                                         (input_ns by default ) Only used if autogather = True
                                   NS_TO_UPDATE:         list of namespaces depending on the scatter namespace
                                                         (by default, we have the list of namespaces of the nested sub_process)   

         |_ DESC_OUT















# Documentation of the vs_MS (very simple Multi Scenarios) driver

## Table of content

- Introduction
- GUI inputs: Selected example
- What is a very simple Multi Scenarios ? What/when do you need it?
	- The SoSTrades general description of a process
	- SoSTrades trades-off or scenarios
	- SoSTrades tree view as disciplines "per scenario nodes" (called as "coupling node")
	- SoSTrades tree view as scenario "per disciplines nodes" (called as "scatter node")
	- SoSTrades tree view as disciplines per scenario nodes with a "gather post-treatment view"
- GUI inputs/outputs
	- GUI inputs: Overview
	- GUI inputs: data input for each instance of the nested sub_process
	- GUI: run
	- GUI outputs: 
	- GUI outputs: Option "autogather" = True
- Builder process: main parameters
- Builder process: how to build a vsMS process?
	- Builder process: Structure of the "Scatter build map" used in the process definition
	- Builder process: About "ns_to_update"
	- Builder process: All possible inputs of  "ns_to_update" in our example and all resulting outcomes
- Structure of Desc_in/Desc_out
- Future evolutions
	- Future evolutions
	- Current small restrictions
- References
- Appendix: from a mathematical point of view

## Introduction
The vs_MS driver is dedicated to create as many scenarios as the user needs to perform on its nested sub_process.

It creates as many instances of the sub_process as selected scenarios. 

Each instance will have its inputs and its outcome outputs (after run).

It provides the capability to:
1. Step 1: Generate all the instances of disciplines based on the user's selected list of scenario names and display it in a tree view. 
2. Step 2: Then allow the user to populate the disciplines
3. Step 3: Run all disciplines 


## GUI inputs: Selected example
The selected example for this manual is a vs_MS applied to the "disc1_disc3" disciplines. There exists a SoSTrades process for this example and it can be selected as follows: 
![Example of driver process](./sos_vs_MS_png/vs_MS_01.PNG)

![Study creation](./sos_vs_MS_png/vs_MS_02.PNG)

![Tree view example](./sos_vs_MS_png/vs_MS_03.PNG)

The user is then ready to provide the inputs depending on his needs.

The user's inputs will be both:
- the selected list of scenario's names ("Scenario List" input) 
- and the inputs to each associated discipline instance of the nested sub_process

The input named as "Scenario List" may have its name customised by the user.

## What is a very simple Multi Scenarios? What/when do you need it?

### General description of a process in SoSTrades
We have a disciple that may be a SoS, i.e. be a structured set of disciplines 'Disk_j'.
It results of a list of input variables 'x_k' that may be:
- either shared (common to all disciplines) and in this case it has an associated namespace key that may have a selected value
- or local to a discipline 'Disc_j'

In this context, each variable will have a unique full name defined as a path in a tree view. 

A variable that has the same full name and belongs to two disciplines with an input/output status, is a coupled variable.



### SoSTrades trades off or scenarios
We want to create several scenarios, where a scenario is a set of 'x_k' coherent input values.
Each process will be defined by its name provided in a list ['Scenario_1', 'Scenario_2', 'Scenario_3', ...]

| Scenario      | Scenario name |               |      'x_j'        |               |               |
| ------------- |:-------------:|:-------------:|:-----------------:|:-------------:|:-------------:|
|      1        | 'Scenario_1'  |               | 'Scenario_1.x_j'  |               |               |
|      2        | 'Scenario_2'  |               | 'Scenario_2.x_j'  |               |               |
|      3        | 'Scenario_3'  |               | 'Scenario_3.x_j'  |               |               |
|               |               |               |                   |               |               |

### SoSTrades tree view as disciplines "per scenario nodes"
The scenario name is then added in the name of the variables and we have the following tree view.

		|_ MyStudyName
		
			|_ vs_MS
			
				|_ scenario_1
				
					|_ Disc_1
					
					|_ Disc_2
					
				|_ scenario_2
				
					|_ Disc_1
					
					|_ Disc_2

In this tree view, 
- scenario node 'scenario_i' (with the scenario name) are added. They are coupling nodes.
- 'vs_MS' is the vs_MS driver node
- 'vs_MS.scenario_i.Disc_j' is an instance of discipline 'Disc_j'
All the disciplines 'vs_MS.scenario_i.Disc_j are children of each scenario nodes 'scenario_i'.

About the input variables:
- if local:  xx
- if shared: xx

About the output variables:
- if local: xx
- if shared: xx

### SoSTrades tree view as scenario "per disciplines nodes" (called as "scatter node")

We can have the same list of scenarios, but with the tree view sorted by disciplines rather than scenarios.
The discipline name with be a "scatter node" driver in the tree, and the scario name be the discipline 'Disc_j' as a children of this scatter driver node: 

		|_ MyStudyName
		
				|_ Disc_1
				
					|_ scenario_1
					
					|_ scenario_2
					
				|_ Disc_2
				
					|_ scenario_1
					
					|_ scenario_2
In this tree view, 
- scatter node 'Disc_j' (with the disc name) are added. It is "scatter nodes".
- 'Disc_j.scenario_i' is an instance of discipline 'Disc_j'
All the disciplines Disc_j.scenario_i are children of each scatter discipline node 'Disc_j'.

this capability is provided by the SoSTrades "scatter driver" and may be depreciated in the future, as we prefer the "per scenario node" view with a "gather postreatment view" (see next section)

### SoSTrades tree view as disciplines per scenario nodes with a "gather postreatment view"

xxx

### "very simple MS" versus "simple MS" versus "MS" versus "Morphological Matrix Eval" drivers
Multi-scenario MS and simple MS are specialisations of "very simple MS" where the list of scenarios are automatically created and populated based on a specified list of inputs with values (see related documentation)

The "Morphological Matrix Eval" xxxx (see related documentation)

## GUI inputs/outputs
### GUI inputs: Overview
 
The overall process appears as a tree view:
- the study name: e.g. Example_vs_MS_Disc1_Disc3
- the driver name: e.g. vs_MS

![Tree view example](./sos_vs_MS_png/vs_MS_03.PNG)

The the user with select a name for each of his scenario:

![Main inputs](./sos_vs_MS_png/vs_MS_04.PNG)

The user needs to select the list of scenario names by using this "Scenario List" selector.

![Eval in](./sos_vs_MS_png/vs_MS_05.PNG)


Remark:
- After configuration, he will then see all the generated instances of disciplines for each selected scenario.


		
- the tree of the nested subprocess: (Disc1,Disc3) with as many instances and names as we have values in the "Scenario List" input.

![Eval in](./sos_vs_MS_png/vs_MS_06.PNG)


The "Scenario List" will usually be placed in the tree view at the vs_MS driver node (however its namescape defined by the "input_ns" value in the "Scatter build map", can be put somewhere else).

There are other parameters associated with the vs_MS driver:
- it can be the "autogather" boolean value but it is hidden in the driver creation process
- it can be the "input_ns" value but it is hidden in the driver creation process
- it can be the "ns_to_update " with the list of namespace that have to be taken into account (to avoid collision of outputs): this list may be computed by SoSTrades in a future evolution and not be anymore a process input


### GUI inputs: data input for each instance of the nested sub_process
The user NEEDs also to provide value for each input variable of each instance of the nested subprocess.
For example, in case of this  example: 
![Eval in](./sos_vs_MS_png/vs_MS_07.PNG)
![Eval in](./sos_vs_MS_png/vs_MS_12.PNG)
![Eval in](./sos_vs_MS_png/vs_MS_08.PNG)
![Eval in](./sos_vs_MS_png/vs_MS_09.PNG)

### GUI: run
When all inputs data have been provided, and saved, the user can run the vs_MS, that goes from the configuration state, to Pending and Done (or Failure with error message if something goes wrong). 

![Eval in](./sos_vs_MS_png/vs_MS_13.PNG)
![Eval in](./sos_vs_MS_png/vs_MS_11.PNG)

### GUI outputs: 
Then the user can see and post-treat the results.
In the output section of each instance of the nested process, he will find the values of each output variable.


Those variables will be put in the treeview in a location that will depend on their namespace.

In our example, the outputs are the following:
![Eval in](./sos_vs_MS_png/vs_MS_10.PNG)

### GUI outputs: Option "autogather" = True

When "autogather" = True in the process, then an output node is added in the tree view with the "gather_node" name provided in the driver process.

All the output values (for all the scenarios) are gathered at this node.

## Builder process: main parameters
- driver_name: it is a string that is set to 'vs_MS' by default. It is the name of the instance of the vs_MS driver
- autogather: it is a boolean that is set to False by default. When set to yes, then gathering of output results are provided in the tree view on an added 'gather node'.
- gather_node: it is a string only used if autogather = True. It provide the name of the 'gather node' in the tree view: i.e. 'Post-processing'
- business_post_proc: boolean that is set to False by default.
- builder_list: it is the class builder of the nested sub_process
- scenario_map_name: it is a string. It is the name of the "Scatter build map" needed to define the management of the list of scenarios and associated namespace of variable and wanted structure. 
See the dedicated section.

## Builder process: how to build a vsMS process?
### Builder process: Structure of the "Scatter build map" used in the process definition
| Key           |  Optional ?   | Description                     |
| ------------- |:-------------:|:-------------------------------:|
| input_name    |  N            | name of the variable to scatter |
| input_type    |  Y            | type of the variable to scatter |
| input_ns      |  Y            | namespace of the variable to scatter |
| output_name   |  Y            | name of the variable to overwrite |
| scatter_ns    |  Y            | namespace associated to the scatter discipline |
| gather_ns     |  Y            | namespace of the gather discipline associated to the scatter discipline (input_ns by default) |
| ns_to_update  |  Y/N          | list of namespaces depending on the scatter namespace|

### Builder process:  About "ns_to_update"
 xxxx

### Builder process:  All possible inputs of  "ns_to_update" in our example and all resulting outcomes


## Structure of Desc_in/Desc_out
Please find here an overview description of the Input/Output widgets tree structure (where Dec_in and desc_out stand for input and output descriptions):

        |_ DESC_IN
                |_  Scenario List (namespace: 'input_ns', structuring)
        |_ DESC_OUT
            |_ 

and a short definition of those DESC parameters:

        Scenario List:                  list of names (one per scenario)

## Future evolutions
### Future evolutions
- The definition of the "Scatter build map" at the process creation step will be simplified

### Current small restrictions
- xxxx

## References

## Appendix: from a mathematical point of view
The vs_MS driver is dedicated to create as many scenarios as have been provided in a defined map:

vsMS:

		List of scenario names -> Design Space of the nested process

		scenario_name |-> point in the design Space

It is realised by creating a dedicated discipline instance per scenario name:

		List of scenario names -> set of discipline objects

		scenario_name |-> a discipline instance with a unique name based on its scenario name


We can then denotes each family by the label of the scenarios:
- the family of scenario inputs $x_{<scenario\_name>})$
- the family of process instances $P_{<scenario\_name>}$
- and the associated outputs by $y_{<scenario\_name>}$, where $$y_{<scenario\_name>} = P_{<scenario\_name>}(x_{<scenario\_name>})$$.

