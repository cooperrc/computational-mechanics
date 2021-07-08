# Computational Mechanics 02 - Analyze Data
## Learning some statistics and data processing skills in Python

Welcome to Computational Mechanics Module #2 - Analyze Data

There are four modules and one final project. The modules will get us started on our exploration of computational
mechanics using Python, listed below each module are the learning objectives. 


1. [01_Cheers_Stats_Beers](./notebooks/01_Cheers_Stats_Beers.ipynb)
  * Read data from a `csv` file using `pandas`.
  * The concepts of Data Frame and Series in `pandas`.
  * Clean null (NaN) values from a Series using `pandas`.
  * Convert a `panda`s Series into a `numpy` array.
  * Compute maximum and minimum, and range.
  * Revise concept of mean value.
  * Compute the variance and standard deviation.
  * Use the mean and standard deviation to understand how the data is distributed
  * Plot frequency distribution diagrams (histograms).
  * Normal distribution and 3-sigma rule.

2. [02_Seeing_Stats](./notebooks/02_Seeing_Stats.ipynb)
  * You should always plot your data.
  * The concepts of quantitative and categorical data.
  * Plotting histograms directly on columns of dataframes, using `pandas`.
  * Computing variance and standard deviation using NumPy built-in functions.
  * The concept of median, and how to compute it with NumPy.
  * Making box plots using `pyplot`.
  * Five statistics of a box plot: the quartiles Q1, Q2 (median) and Q3 (and
  * interquartile range Q3$-$Q1), upper and lower extremes.
  * Visualizing categorical data with bar plots.
  * Visualizing multiple data with scatter plots and bubble charts.
  * `pandas` is awesome!

3. [03_Linear_Regression_with_Real_Data](./notebooks/03_Linear_Regression_with_Real_Data.ipynb)
  * Making our plots more beautiful
  * Defining and calling custom Python functions
  * Applying linear regression to data
  * NumPy built-ins for linear regression
  * The Earth is warming up!!!

4. [04_Stats_and_Montecarlo](./notebooks/04_Stats_and_Montecarlo.ipynb)
  - How to generate "random" numbers in Python+
  - The definition of a Monte Carlo model
  - How to calculate $\pi$ with Monte Carlo
  - How to take the integral of a function with Monte Carlo
  - How to propagate uncertainty in a model with Monte Carlo
  - **Bonus**: use Sympy to do calculus and algebra for us! _no need for
  - Wolfram, sorry Stephen_
  - How to generate a normal distribution using uniformly random numbers

  +The computer only generates pseudo-random numbers. For further
  information **and** truly random numbers  check
  [www.random.org](https://www.random.org/randomness/) 

## [Computational Mechanics Project #02 - Create specifications for a spitballing robot](./project/02_Analyze-data_project.ipynb)

On the first day of class, we threw $2"\times~2"$ dampened paper (spitballs) at
a target on the whiteboard. Now, we are going to analyze the accuracy of the
class with some cool Python tools and design a robot that has the same accuracy
and precision as the class. 

The goal of this project is to determine the precision of necessary components
for a robot that can reproduce the class throwing distibution. We have generated
pseudo random numbers using `numpy.random`, but the class target practice is an
example of truly random distributions. If we repeated the exercise, there is a
vanishingly small probability that we would hit the same points on the target,
and there are no deterministic models that could take into account all of the
factors that affected each hit on the board. 

<img src="./images/robot_design.png" style="height: 250px;"/>

