# Computational Mechanics 01 - Getting Started
## Working with Python and Numerical Methods

Welcome to Computational Mechanics Module #1 - Getting Started!

There are three modules and one final project. The modules will get us started on our exploration of computational
mechanics using Python, listed below each module are the learning objectives. 

## [01_Interacting_with_Python](./01_Interacting_with_Python.md)
  * Using the `print()` function. The concept of _function_.
  * Using Python as a calculator.
  * Concepts of variable, type, assignment.
  * Special variables: `True`, `False`, `None`.
  * Supported operations, logical operations. 
  * Reading error messages.

## [02_Working_with_Python](./02_Working_with_Python.md)
  * Good coding habits and file naming
  * How to define a function and return outputs
  * How to import libraries
  * Multidimensional arrays using NumPy
  * Accessing values and slicing in NumPy arrays
  * `%%time` magic to time cell execution.
  * Performance comparison: lists vs NumPy arrays
  * Basic plotting with `pyplot`.

## [03_Numerical_error](./03_Numerical_error.md)
  * Numerical integration with the Euler approximation
  * The source of truncation errors
  * The source of roundoff errors
  * How to time a numerical solution or a function
  * How to compare solutions
  * The definition of absolute error and relative error
  * How a numerical solution converges

## [HW_01](./HW_01.md)

## [Computational Mechanics Project #01 - Heat Transfer in Forensic Science](../projects/01_Getting-started-project.md)

We can use our current skillset for a macabre application. We can predict the
time of death based upon the current temperature and change in temperature of a
corpse. 

Forensic scientists use Newton's law of cooling to determine the time elapsed
since the loss of life, 

$\frac{dT}{dt} = -K(T-T_a)$,

