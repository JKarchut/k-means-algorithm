# K Means Algortithm

2 example implementation of k means clustering algorithm using GPU CUDA toolkit

## Problem

Given N points of n dimenstions find k clusters that would best cover the given points such that each point belongs to closest cluster. Since this is NP-hard problem algorithm is heuristics.

## Solution 

Algorith in each iteration finds closes cluster for each point (initially clusters will be chosen as first k points). Then for each cluster their center is recalculated as average value of belonging points.
Computation reapeats until no point changes it's assigned cluster.

## Lunching program

./run.sh launches both implementations of algorithm and compares their computation time (first provide test.txt file using fileGenerator program provided in Helpers folder).

## Algorithm implementation

The hardest part of implementing this algotihm using CUDA is to effectively calculate new clusters center and two solutions to this problem were provided. First uses standard reduce algorithm, secound one before calculating new centers sorts the dataset based on the membership of a point to a certaing clusted and then runs reduce.