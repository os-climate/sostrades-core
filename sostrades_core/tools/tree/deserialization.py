import ast
import numpy as np


def isevaluatable(s):
    """
    Check if string only contains a literal of type - strings, numbers, tuples, lists, dicts, booleans, and None
    :param s:
    :return:
    """
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError) as e:
        # deal with numpy arrays that have the numpy format (convert it to list then to array)
        if isinstance(s, str) and s.startswith('[') and s.endswith(']'):
            return evaluate_arrays(s)
        return s

def evaluate_arrays(input_str):
    """
    convert a string into an array or a list of array
    :param input_str: the string to convert into an array
    :type string
    :return: the numpy array
    """
    # fix the \n and , if needed and split by ' '
    array_content = input_str.replace('array(' ,'').replace(')' ,'').replace('\n' ,' ').replace(',' ,' ').split(' ')
    # remove empty entry
    array_content = [x for x in array_content if x != '']
    # check bracket alone (when there is a space between bracket and digit '[ 1 2 ]'
    # we need to remove the bracket alone and add it to the next digit)
    for i in range(0 ,len(array_content)):
        if array_content[i] == '[' and i+ 1 < len(array_content):
            array_content[i + 1] = '[' + array_content[i + 1]
        if array_content[i] == ']' and i - 1 >= 0:
            array_content[i - 1] = array_content[i - 1] + ']'
    array_content = [x for x in array_content if x != '[' and x != ']']
    # recreate the string list that can be interpreted as a list
    new_s = ','.join(array_content)
    try:
        # convert the string in list then in arrays
        eval = convert_list_to_arrays(ast.literal_eval(new_s))
        # the writing of an array into a list if array() instead of [x y]
        if 'array(' in input_str:
            return list(eval)
        else:
            return eval
    except Exception as e:
        return input_str


def convert_list_to_arrays(input_list):
    """
    convert a list into an array and if the list contains list, convert into array of arrays
    :param input_list: the list to convert into an array
    :type list
    :return: the list converted into numpy array
    """
    if isinstance(input_list, list):
        # Si la liste contient d'autres listes, récursion
        return np.array([convert_list_to_arrays(item) for item in input_list])
    else:
        # Si l'élément est un nombre return the element
        return input_list

