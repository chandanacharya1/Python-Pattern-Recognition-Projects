Fit a bi-variate Gaussian to the height and weight data in x and y to model
the joint density p(x, y) of heights and weights.

Given your fitted bi-variate Gaussian, use the idea of conditional expecta-
tion to predict the weight values for the outliers. That is, let xo denote the

available height data of an outlier and compute
E[y | xo] = Integral{y p (y | xo)dy}

Do this either analytically as discussed in the lecture or numerically and
report your results.