#!/usr/bin/env python

import functools, operator

# Copied from StackOverflow response
def getFromDict(data, path):
    return functools.reduce(operator.getitem, path, data)

# Copied from StackOverflow response
def setInDict(data, path, val):
    getFromDict(data, path[:-1])[path[-1]] = val