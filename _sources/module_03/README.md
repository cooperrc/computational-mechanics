# Computational Mechanics 03- Initial Value Problems
## Learning to frame engineering equations as numerical methods

Welcome to Computational Mechanics Module #3! In this module we will explore
some more data analysis, find better ways to solve differential equations, and
learn how to solve engineering problems with Python. 

[01_Catch_Motion](./notebooks/01_Catch_Motion.ipynb)

* Work with images and videos in Python using `imageio`.
* Get interactive figures using the `%matplotlib notebook` command.
* Capture mouse clicks with Matplotlib's `mpl_connect()`.
* Observed acceleration of falling bodies is less than $9.8\rm{m/s}^2$.
* Capture mouse clicks on several video frames using widgets!
* Projectile motion is like falling under gravity, plus a horizontal velocity.
* Save our hard work as a numpy .npz file __Check the Problems for loading it back into your session__
* Compute numerical derivatives using differences via array slicing.
* Real data shows free-fall acceleration decreases in magnitude from  $9.8\rm{m/s}^2$.

[02_Step_Future](./notebooks/02_Step_Future.ipynb)

* Integrating an equation of motion numerically.
* Drawing multiple plots in one figure,
* Solving initial-value problems numerically
* Using Euler's method.
* Euler's method is a first-order method.
* Freefall with air resistance is a more realistic model.

[03_Get_Oscillations](./notebooks/03_Get_Oscillations.ipynb)

* vector form of the spring-mass differential equation
* Euler's method produces unphysical amplitude growth in oscillatory systems
* the Euler-Cromer method fixes the amplitude growth (while still being first
* order)
* Euler-Cromer does show a phase lag after a long simulation
* a convergence plot confirms the first-order accuracy of Euler's method
* a convergence plot shows that modified Euler's method, using the derivatives
* evaluated at the midpoint of the time interval, is a second-order method
* How to create an implicit integration method
* The difference between _implicit_ and _explicit_ integration
* The difference between stable and unstable methods

[04_Getting_to_the_root](./notebooks/04_Getting_to_the_root.ipynb)

* How to find the 0 of a function, aka root-finding
* The difference between a bracketing and an open methods for finding roots
* Two bracketing methods: incremental search and bisection methods
* Two open methods: Newton-Raphson and modified secant methods
* How to measure relative error
* How to compare root-finding methods
* How to frame an engineering problem as a root-finding problem
* Solve an initial value problem with missing initial conditions (the shooting
* method)
* _Bonus: In the Problems you'll consider stability of bracketing and open  methods._
