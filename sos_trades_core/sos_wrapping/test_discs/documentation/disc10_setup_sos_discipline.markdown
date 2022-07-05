# The Disc10 discipline is a dynamic discipline that switches between 3 models.

If Model_Type== 'Linear':

	y = Disc10(a,x) = a * x

elif Model_Type== 'Affine':

	y = Disc10(a,x,b) = a * x + b

elif Model_Type== 'Polynomial':

	y = Disc10(a,x,b,power) =  a * x**power + b
	
	
Default value of 'a' is 1.

Default value of 'power' is 2.
