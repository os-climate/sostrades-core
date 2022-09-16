# Documentation for DoE_eval driver (process builder)


## Short description
The build doe_eval discipline driver is a specialisation of the doe_eval driver. 
It extends the doe_eval discipline driver by providing the capability to select in the GUI the used sub_process of the driver. 

The doe_eval capability provides or creates a sample and to run the subprocess on this sample. 


## Structure of Desc_in/Desc_out
        |_ DESC_IN
            |_ SUB_PROCESS_INPUTS (structuring)
                |_ EVAL_INPUTS (namespace: 'ns_doe_eval', structuring, dynamic : self.sub_proc_build_status != 'Empty_SP') NB: Mandatory not to be empty (If not then warning)
                |_ EVAL_OUTPUTS (namespace: 'ns_doe_eval', structuring, dynamic : self.sub_proc_build_status != 'Empty_SP') NB: Mandatory not to be empty (If not then warning)
                |_ SAMPLING_ALGO (structuring, dynamic : self.sub_proc_build_status != 'Empty_SP')
                        |_ CUSTOM_SAMPLES_DF (dynamic: SAMPLING_ALGO=="CustomDOE") NB: default DESIGN_SPACE depends on EVAL_INPUTS (As to be "Not empty") And Algo 
                        |_ DESIGN_SPACE (dynamic: SAMPLING_ALGO!="CustomDOE") NB: default DESIGN_SPACE depends on EVAL_INPUTS (As to be "Not empty") And Algo
                        |_ ALGO_OPTIONS (structuring, dynamic: SAMPLING_ALGO != None)
                        |_ <var multiplier name>: for each selected input with MULTIPLIER_PARTICULE in its name
            |_ N_PROCESSES
            |_ WAIT_TIME_BETWEEN_FORK
            |_ NS_IN_DF (dynamic: if sub_process_ns_in_build is not None)
        |_ DESC_OUT
            |_ SAMPLES_INPUTS_DF (namespace: 'ns_doe_eval')
            |_ <var>_dict (internal namespace 'ns_doe', dynamic: sampling_algo!='None' and eval_inputs not empty and eval_outputs not empty, for <var> in eval_outputs)

##     Description of DESC parameters
        SUB_PROCESS_INPUTS: 	    All inputs for driver builder in the form of a dictionary of four keys        
            'PROCESS_REPOSITORY':       folder root of the sub processes to be nested inside the DoE.
                                        If 'None' then it uses the sos_processes python for doe creation.
            'PROCESS_NAME':             selected process folder name to be nested inside the DoE.
                                        If 'None' then it uses the sos_processes python for doe creation.
            'PROCESS_NAME' :            either empty or an available usecase of the sub_process
            'USECASE_DATA' :            anonymized dictionary of usecase inputs to be nested in context
                                        it is a temporary input: it will be put to None as soon as                                                                        
                                        its content is 'loaded' in the dm. We will have it has editable    
        EVAL_INPUTS:                    selection of input variables to be used for the DoE
        EVAL_OUTPUTS:                   selection of output variables to be used for the DoE (the selected observables)
        SAMPLING_ALGO:                  method of defining the sampling input dataset for the variable chosen in self.EVAL_INPUTS
        N_PROCESSES:
        WAIT_TIME_BETWEEN_FORK:
        SAMPLES_INPUTS_DF :             copy of the generated or provided input sample
        ALL_NS_DICT :                   a map of ns keys: values
        CUSTOM_SAMPLES_DF:              provided input sample
        DESIGN_SPACE:                   provided design space
        ALGO_OPTIONS:                   options depending of the choice of self.SAMPLING_ALGO
        <var observable name>_dict':    for each selected output observable doe result
                                        associated to sample and the selected observable

## Example of models to be nested

### csv files for sub_process_inputs
In the "empty driver process" folder, csv files for sub_process_inputs are provided for a list of nested models:

![sub_process_inputs](./doe_eval_png/DoE_Eval_24.PNG)

### Models in usecases
In the "empty driver process" folder, complete uscases are provided for a list of nested models:

![Study creation](./doe_eval_png/DoE_Eval_25.PNG)

## The several usage steps are detailed below.

###  Step 1: xxxxx

xxxx

## Future evolutions
- The input selection of the process/associated usecase will be replaced by a dedicated widget in SoSTrades GUI

## Current small restrictions
- The driver name is 'DoE_Eval' and cannot be changed by the user
- The namespace 'ns_doe_eval' is set to f'{self.ee.study_name}.DoE_Eval'


