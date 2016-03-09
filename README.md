# TinyTimeModels

[![Build Status](https://travis-ci.org/GordStephen/TinyTimeModels.jl.svg?branch=master)](https://travis-ci.org/GordStephen/TinyTimeModels.jl)

A Julia package for fitting univariate local-level structural time series models, with diffuse state initialization and optional linear regression terms.

`fit(y::Vector)` fits an observed time series (with missing values allowed as `NaN`).

`fit(y::Vector, X::Matrix)` fits an observed time series (with missing values allowed as `NaN`) and estimates regression coefficients for an arbitrary number of contemporaneous series (no missing data allowed).

For examples, see the [tests](https://github.com/GordStephen/TinyTimeModels.jl/blob/master/test/runtests.jl).
