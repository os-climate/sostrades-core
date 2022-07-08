# The Combvec discipline

DESC_IN

	- my_dict_of_vec: {'type': 'dict', 'subtype_descriptor': {'dict': 'float'}, 'default': default_my_dict_of_vec, 'unit': '-', 'visibility': SoSDiscipline.LOCAL_VISIBILITY}

DESC_OUT

	- custom_samples_df: {'type': 'dataframe', 'unit': '-', 'visibility': SoSDiscipline.SHARED_VISIBILITY,'namespace': 'ns_doe_eval'}
RUN

If my_dict_of_vec is:

    - my_dict_of_vec['stat_A'] = [2, 7]
    - my_dict_of_vec['stat_B'] = [2]
    - my_dict_of_vec['stat_C'] = [3, 4, 8]
	
then custom_samples_df is a dataframe with headers ['stat_A','stat_B','stat_C'] and each row is all possible combinations of ('stat_A' value,'stat_B' value,'stat_C' value) based on my_dict_of_vec selected values.

Here custom_samples_df is :

	- ['stat_A','stat_B','stat_C'] 
	- 2 2 3
	- 2 2 4
	- 2 2 8
	- 7 2 3
	- 7 2 4
	- 7 2 8
	
	
