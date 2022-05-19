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

# How to use a `for` loop? _in Python_

The `for` loop has 2 main components:
1. an iterating variable e.g. `i`
2. a list of desired values e.g. `range(3)` will be three values `i = 0, i = 1, i = 2`

These loops are _great_ when you want to repeat the same command multiple times. You can use the variable to update your command. 

Consider this `for` loop, the iterating variable is `i` and the desired values are `range(3)`:

```{code-cell} ipython3
for i in range(3):
    print('i =', i)
```

The `for` loop reassigns a variable e.g. `i` each time it finishes the commands that are tabbed. 

```
for <variable> in <desired_values
```
$\rightarrow$ `    <run these commands>`

+++

You can accomplish the same output without a `for`-loop by copy-pasting the `print` command:

```{code-cell} ipython3
print('i = ', 0)
print('i = ', 1)
print('i = ', 2)
```

## Use a `for` loop to calculate a sum

The [`np.sum`](https://numpy.org/doc/stable/reference/generated/numpy.sum.html) will sum the values of an array, but if you have a Python list _inside `[` and `]`_ it will return an error. You _can_ use a `for` loop to calculate this value (_if possible use NumPy though!_). Here is the process:

1. define the vector, `v`
2. initialize the value for your sum, `sum_v`
3. create a `for`-loop that assigns your variable to each value of `v`, `for vi in v:`
4. inside the `for`-loop, add each value to `sum_v`, `sum_v += vi`

```{code-cell} ipython3
v = [1, 2, 3, 10]

sum_v = 0

for vi in v:
    sum_v += vi
print('sum of vectors components in v is', sum_v)
```

## Wrapping up

In this notebook, you built some `for`-loops to `print` and add variables from desired values. You can use the `for` loops to repeat the same command multiple times while changing the the iterating variable.
