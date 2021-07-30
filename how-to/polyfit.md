---
jupytext:
  formats: notebooks/ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.10.3
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from numpy.random import default_rng
```

# Creating a quadratic fit on random data

In this notebook, you will 
- build a variable of fabricated falling heights, `y` = $\frac{g}{2}t^2 + error$
- find the best-fit quadratic function with `np.polyfit`
- plot the data and fit using `poly1d`

```{code-cell} ipython3
rng = default_rng()
x = np.linspace(0,5)
y = 9.81/2*x**2 + (rng.random(len(x)) - 0.5)*4

plt.plot(x, y)
```

Above, the fabricated falling data is shown for 5 seconds. The error is introduced as uniformly random numbers from -2 - 2 as `(rng.random(len(x) - 0.5)*4`. 

Next, you build the polynomial fit with `np.polyfit`. Finally, plug the polynomial constants into the `np.poly1d` function to created a function for $y_{fit}(x)$ = `pp(x)`. 

```{code-cell} ipython3
A = np.polyfit(x, y, 2)
pp = np.poly1d(A)
plt.plot(x, y,'o', label = 'data')
plt.plot(x, pp(x), label = 'quadratic fit')
plt.legend();
```
