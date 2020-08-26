# Chaotic Robot

A cleaning robot will cover more surface being programmed to move following chaotic equation of motion or a random walk?
Study Arnold's flow we can answer.

Project by Silo Cui and Markus Ernstsson, [report](./Arnold_flow.pdf) in `Arnold_flow.pdf`

## Content

Arnoldtraj.py. Calculates Arnold's flow trajectory, adjust `traj`, `poin`, `lya_exp` to `True` to calculate 3D trajectory, Poincare' section, and Lyapunov exponent respectively.

## Reproducibility

Run with `python3` with
```
scipy (1.1.0)
numpy (1.17.2)
matplotlib (1.5.1)
```
packages.