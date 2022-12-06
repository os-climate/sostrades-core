## Documentation

Function manager is a model that aggregates weighted objectives, inequality and equality constraints using smoothing functions.
It takes both dataframes/arrays and floats as objectives/constraints. 

![](scheme_funcmanager.png)

### Scalarize functions 

Objectives, inequality constraints and equality constraints functions enter the function manager with a numpy array containing a single (objective) or multiple (constraints) values.
The first step towards the aggregation of these functions is to merge these numpy array into a single scalar value.

Each type of function is treated individually at first:

	-objectives:
		the values are either summed or the smooth_maximum function is applied on value, depending on the 'aggr_type' variable of the function.
	-equality and inequality constraints:
		the array goes through a function 'cst_func_smooth_positive' to scale the values between 0 and +inf, then the smooth_maximum value is returned. 
		
The scalar value of each function is then multiplied by the weight associated with the function

#### Function 'smooth\_maximum'

The 'smooth\_maximum' function is applied on arrays of values, to smooth the transition between functions and ensure continuity.

It uses the same principal as in the Kreisselmeier-Steinhauser function [^1].

$$max_{exp} = 650$$
$$max_{\alpha x} = max(\alpha * cst)$$
$$k = max_{\alpha x} - max_{exp}$$
$$result=\frac{\sum values . e^{\alpha * cst - k}}{\sum e^{\alpha * cst - k}}$$

By default $$\alpha$$ is set to 3

![](smooth_maximum.png)


#### Function 'cst\_func\_smooth\_positive'

This function loops on the numpy array and apply one of four functions on each entry (f(value)=res):

	- if value < -250.0 : res = 0.0
	- if -250.0 < val < eps : res = eps * (np.exp(val) - 1.)
	- if val.real > eps: res = (1. - eps / 2) * val ** 2 + eps * val
	- if self.smooth_log and val.real > self.eps2: res = res0 + 2 * np.log(val)

The four parts of this function are designed to ensure a smooth continuous function, with a quadratic increase above 'eps'.
If 'smoothlog' is set to True, the function is capped with a log increase above eps2, to avoid some numeric issues with constraints 
values going haywire.
!['Continuous positive smoothing function'](residuals_wo_smoothlog.png)

### Aggregate functions

The scalarized functions are then aggregated, first into three functions aggregating each type of functions, and then into a single objective used to drive the optimization: the lagrangian objective.

It is calculated as the sum of the three aggregated functions multiplied by 100.


[^1]: [Martins, J. R. R. A., and Nicholas MK Poon. "On structural optimization using constraint aggregation." VI World Congress on Structural and Multidisciplinary Optimization WCSMO6, Rio de Janeiro, Brasil. 2005.](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.140.3612&rep=rep1&type=pdf)


