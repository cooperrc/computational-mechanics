# Computational Mechanics 4 - Linear Algebra

Welcome to Computational Mechanics Module #4! In this module we will explore
applied linear algebra for engineering problems and revisit the topic of linear
regression with a new toolbox of linear algebra. Our main goal, is to transform
large systems of equations into manageable engineering solutions. 

## [01_Linear-Algebra](./01_Linear-Algebra.md)


* How to solve a linear algebra problem with `np.linalg.solve`
* Creating a linear system of equations
* Identify constants in a linear system $\mathbf{A}$ and $\mathbf{b}$
* Identify unknown variables in a linear system $\mathbf{x}$
* Identify a __singular__ or __ill-conditioned__ matrix
* Calculate the __condition__ of a matrix
* Estimate the error in the solution based upon the condition of a matrix

## [02_Gauss_elimination](./02_Gauss_elimination.md)


* Graph 2D and 3D linear algebra problems to identify a solution (intersections
* of lines and planes)
* How to solve a linear algebra problem using __Gaussian elimination__ (`GaussNaive`)
* Store a matrix with an efficient structure __LU decomposition__ where  $\mathbf{A=LU}$
* Solve for $\mathbf{x}$ using forward and backward substitution (`solveLU`)
* Create the __LU Decomposition__ using the Naive Gaussian elimination process (`LUNaive`)
* Why partial __pivoting__ is necessary in solving linear algebra problems
* How to use the existing `scipy.linalg.lu` to create the __PLU decomposition__
* How to use the __PLU__ efficient structure to solve our linear algebra problem (`solveLU`)

## [03_Linear-regression-algebra](./03_Linear-regression-algebra.md)

* How to use the _general least squares regression_ method for almost any function
* How to calculate the coefficient of determination and correlation coefficient for a general least squares regression, $r^2~ and~ r$
* Why we need to avoid __overfitting__
* How to construct general least squares regression using the dependent and independent data to form $\mathbf{y}=\mathbf{Za}$. 
* How to construct a piecewise linear regression 

## [HW_04](./HW_04.md)


## [Project_04 - Practical Linear Algebra for Finite Element Analysis](../projects/04_Linear_Algebra-project.md)

In this project we will perform a linear-elastic finite element analysis (FEA) on a support structure made of 11 beams that are riveted in 7 locations to create a truss as shown in the image below. 

![Mesh image of truss](../images/mesh.png)

+++

The triangular truss shown above can be modeled using a [direct stiffness method [1]](https://en.wikipedia.org/wiki/Direct_stiffness_method), that is detailed in the [extra-FEA_material](./extra-FEA_material.ipynb) notebook. The end result of converting this structure to a FE model. Is that each joint, labeled $n~1-7$, short for _node 1-7_ can move in the x- and y-directions, but causes a force modeled with Hooke's law. Each beam labeled $el~1-11$, short for _element 1-11_, contributes to the stiffness of the structure. We have 14 equations where the sum of the components of forces = 0, represented by the equation


