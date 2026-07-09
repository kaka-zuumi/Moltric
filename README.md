
For installation of Moltric, use pip: `pip install moltric`

For questions or suggestions for features to add, contact Kazuumi Fujioka: kazuumi@hawaii.edu


# Introduction

<img align="right" width="600" height="300" src="permutation1.png">

<p align="justify">
Blah blah blah...
</p>

Consider the reaction below:

$$
\textrm{CH} + \textrm{C}_4 \textrm{H}_6 \longrightarrow \textrm{C}_5 \textrm{H}_6 + \textrm{H}
$$



## For terminal/coding use:

From a terminal, using the command line interface (CLI) is usually easier. First, create a PAN input file that describes your experimental setup and data, like so:

```
moltric/moltric.py examples/fromFujioka_CH.indene_19atoms.xyz examples/fromFujioka_CH.indene_19atoms.xyz
```

this produces:

```
# Number of points found in 'examples/fromFujioka_CH.indene_19atoms.xyz': 81
# Number of points found in 'examples/fromFujioka_CH.indene_19atoms.xyz': 81
#    i    j      DMD
/home/kazuumi/.conda/envs/sgdmlSTUFF/lib/python3.11/site-packages/ot/bregman/_sinkhorn.py:666: UserWarning: Sinkhorn did not converge. You might want to increase the number of iterations `numItermax` or the regularization parameter `reg`.
  warnings.warn(
     0    0      0.0000
     0    1      2.1084
     0    2      3.9254
     0    3      3.9358
     0    4      1.7824
     0    5      2.6834
     0    6      5.1325
     0    7      3.6761
     0    8      6.9604
     0    9      2.2738
.
.
.
```


