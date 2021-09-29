# How to build a normal distribution with uniformly random numbers?

"Everybody believes in the exponential law of errors: the experimenters, because they think it can be proved by mathematics; and the mathematicians, because they believe it has been established by observation" [5].

In the previous example, we drew dimensions from uniformly random distributions and normally distributed random distributions. Why do we use the "normal" distribution to describe data with a mean and standard deviation? There are exact statistical methods to derive the normal distribution, but let's take a look at a Monte Carlo approach. 

Let's say there are 10 different independent factors that could change the dimensions of the steel bar in question e.g. which tool was used, how old the blade is, the humidity, the temperature, and the list goes on. 

Let's consider one dimension. 
Each of these factors could change the dimensions of the part, let's use a uniform scale of -1/2-1/2.
If the effect is 0, the dimension is exactly as specified. If the effect is -1/2, the dimension is much smaller. Conversely, if the effect is 1/2 the dimension is much larger. Now, we use a Monte Carlo model to generate 10 effects on 10,000 parts as shown in the next block.

```{code-cell} ipython3
factors = np.random.rand(10000,10)-1/2 # each row represents a part and each column is an effect (-1/2-1/2)
```

Now, we have created 10,000 parts with 10 uniformly random effects between -1/2-1/2. 

We sum the effects and look at the final part distribution. The x-axis is labeled "A.U." for arbitrary units, we are just assuming an effect of -1/2-1/2 for each of the 10 factors.  

```{code-cell} ipython3
dims = np.sum(factors,axis=1)

plt.hist(dims,30)
plt.xlabel('effect A.U.')
plt.ylabel('number of parts')
```

Now, depending upon which random numbers were generated, you should see what looks like a normal distribution. 

Normal distributions come from the assumption that we have a large (or
infinite) number of uncontrollable factors that can change our desired result.
In our case, ideally each factor would have an effect of 0, because then it is
exactly as specified, but the reality is that we can't control most factors. As
engineers, we always have to consider the uncertainty in our models and
measurements. 
