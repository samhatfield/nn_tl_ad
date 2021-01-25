# Tangent-linear and adjoint test of a neural network

This repository contains a fully-connected neural network implemented in Fortran, along with its
"tangent-linear" and "adjoint" versions (i.e. the Jacobian and transpose of the Jacobian). It also
includes standard tests for the consistency of these.

This code supports the publication Hatfield, Dueben, Lopez, Geer, Chantry and Palmer (2021).

## How to build
1. Set the environment variable to the location of your BLAS library, e.g. `export LIBBLAS=/usr/lib64`.
2. Run `make`.

## How to run
1. Run `./main`. You should see something like this:
```
Tangent-linear test
Size of perturbation ~=   1.0E+00
Difference of 1st element around digit number   2

Comparison of NL(x) from TL and NL models (should always be identical):
  1.10974418896988292  1.13884408794442193 -3.33824393953981469  0.22562426325598517  1.79357412195060317
  1.10974418896988292  1.13884408794442193 -3.33824393953981469  0.22562426325598517  1.79357412195060317

Size of perturbation ~=   1.0E-01
Difference of 1st element around digit number   3

Comparison of NL(x) from TL and NL models (should always be identical):
  1.10974418896988292  1.13884408794442193 -3.33824393953981469  0.22562426325598517  1.79357412195060317
  1.10974418896988292  1.13884408794442193 -3.33824393953981469  0.22562426325598517  1.79357412195060317

Size of perturbation ~=   1.0E-02
Difference of 1st element around digit number   5

Comparison of NL(x) from TL and NL models (should always be identical):
  1.10974418896988292  1.13884408794442193 -3.33824393953981469  0.22562426325598517  1.79357412195060317
  1.10974418896988292  1.13884408794442193 -3.33824393953981469  0.22562426325598517  1.79357412195060317

Size of perturbation ~=   1.0E-03
Difference of 1st element around digit number   7

Comparison of NL(x) from TL and NL models (should always be identical):
  1.10974418896988292  1.13884408794442193 -3.33824393953981469  0.22562426325598517  1.79357412195060317
  1.10974418896988292  1.13884408794442193 -3.33824393953981469  0.22562426325598517  1.79357412195060317

Size of perturbation ~=   1.0E-04
Difference of 1st element around digit number   9

Comparison of NL(x) from TL and NL models (should always be identical):
  1.10974418896988292  1.13884408794442193 -3.33824393953981469  0.22562426325598517  1.79357412195060317
  1.10974418896988292  1.13884408794442193 -3.33824393953981469  0.22562426325598517  1.79357412195060317

Size of perturbation ~=   1.0E-05
Difference of 1st element around digit number  11

Comparison of NL(x) from TL and NL models (should always be identical):
  1.10974418896988292  1.13884408794442193 -3.33824393953981469  0.22562426325598517  1.79357412195060317
  1.10974418896988292  1.13884408794442193 -3.33824393953981469  0.22562426325598517  1.79357412195060317

Size of perturbation ~=   1.0E-06
Difference of 1st element around digit number  13

Comparison of NL(x) from TL and NL models (should always be identical):
  1.10974418896988292  1.13884408794442193 -3.33824393953981469  0.22562426325598517  1.79357412195060317
  1.10974418896988292  1.13884408794442193 -3.33824393953981469  0.22562426325598517  1.79357412195060317

Size of perturbation ~=   1.0E-07
Difference of 1st element around digit number  16

Comparison of NL(x) from TL and NL models (should always be identical):
  1.10974418896988292  1.13884408794442193 -3.33824393953981469  0.22562426325598517  1.79357412195060317
  1.10974418896988292  1.13884408794442193 -3.33824393953981469  0.22562426325598517  1.79357412195060317

Size of perturbation ~=   1.0E-08
Difference of 1st element around digit number  16

Comparison of NL(x) from TL and NL models (should always be identical):
  1.10974418896988292  1.13884408794442193 -3.33824393953981469  0.22562426325598517  1.79357412195060317
  1.10974418896988292  1.13884408794442193 -3.33824393953981469  0.22562426325598517  1.79357412195060317

Size of perturbation ~=   1.0E-09
Difference of 1st element around digit number  16

Comparison of NL(x) from TL and NL models (should always be identical):
  1.10974418896988292  1.13884408794442193 -3.33824393953981469  0.22562426325598517  1.79357412195060317
  1.10974418896988292  1.13884408794442193 -3.33824393953981469  0.22562426325598517  1.79357412195060317

Size of perturbation ~=   1.0E-10
Difference of 1st element around digit number  16

Comparison of NL(x) from TL and NL models (should always be identical):
  1.10974418896988292  1.13884408794442193 -3.33824393953981469  0.22562426325598517  1.79357412195060317
  1.10974418896988292  1.13884408794442193 -3.33824393953981469  0.22562426325598517  1.79357412195060317

-----------------------------------------------------

Adjoint test
The following two numbers must be as similar as possible
LHS  0.00074264770005028
RHS  0.00074264770005028
Difference of 1st element around digit number   0
```

