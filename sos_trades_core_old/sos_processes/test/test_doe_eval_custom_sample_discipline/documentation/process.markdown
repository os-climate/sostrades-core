# Example and test of a discipline that generates a custom sample

Test of a discipline "combvect" that generates the custom sample "custom_samples_df". 

This discipline is chained with a custom DoE_eval by its coupled variable "custom_samples_df". 

The DoE_eval is applied to a "sumstat" discipline that is doing the sum of three variables. 

The "custom_samples_df" is a local variable of the DoE_eval discipline and is a shared output of "combvect". 
The path of this output variable is selected (through the value of the namespace of this variable) to be the same as the path of the local variable of the DoE-Eval discipline so as to perform the coupling.

In this context of DoE, the sumsat discipline will perform the sum of the 3 columns of the "custom_samples_df". 
