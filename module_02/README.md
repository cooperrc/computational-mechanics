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
  * How to generate "random" numbers in Python$^+$
  * The definition of a Monte Carlo model
  * How to calculate $\pi$ with Monte Carlo
  * How to model Brownian motion with Monte Carlo
  * How to propagate uncertainty in a model with Monte Carlo

  $^+$ Remember, the computer only generates pseudo-random numbers. For
  further information **and** truly random numbers  check
  [www.random.org](https://www.random.org/randomness/) 

## [Project #02 - NYSE random walk predictor](../projects/02_Analyze-data_project)

In the [Stats and Monte Carlo](../module_02/04_Stats_and_Montecarlo) module, you created a Brownian motion model to predict the motion of particles in a fluid. The Monte Carlo model took steps in the x- and y-directions with random magnitudes. 

This [random walk](https://en.wikipedia.org/wiki/Random_walk_hypothesis) can be used to predict stock prices. Let's take a look at some data from the New York Stock Exchange [NYSE](https://www.kaggle.com/dgawlik/nyse) from 2010 through 2017. 

> __Important Note__: 
> I am not a financial advisor and these models are _purely_ for academic exercises. If you decide to use anything in these notebooks to make financial decisions, it is _at your own risk_. _I am not an economist/financial advisor/etc., I am just a Professor who likes to learn and exeriment._

Here, I will show an example workflow to analyze and predict the Google
stock price [[GOOGL]](https://en.wikipedia.org/wiki/Alphabet_Inc.) from
2010 - 2014. Then, you can choose your own stock price to evaluate and
create a predictive model. 

1. Explore data and select data of interest
2. Find statistical description of data: mean and standard deviation
3. Create random variables
4. Generate random walk for [[GOOGL]](https://en.wikipedia.org/wiki/Alphabet_Inc.) stock opening price
