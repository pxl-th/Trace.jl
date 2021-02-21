# Initial benchmark.

BenchmarkTools.Trial:
  memory estimate:  2.63 GiB
  allocs estimate:  57261134
  --------------
  minimum time:     5.272 s (16.08% GC)
  median time:      5.272 s (16.08% GC)
  mean time:        5.272 s (16.08% GC)
  maximum time:     5.272 s (16.08% GC)
  --------------
  samples:          1
  evals/sample:     1

# Change RGBSpectrum to use Point3f0 instead of Vector.

BenchmarkTools.Trial:
  memory estimate:  2.60 GiB
  allocs estimate:  56999570
  --------------
  minimum time:     5.415 s (15.65% GC)
  median time:      5.415 s (15.65% GC)
  mean time:        5.415 s (15.65% GC)
  maximum time:     5.415 s (15.65% GC)
  --------------
  samples:          1
  evals/sample:     1

# Tweaks to hashing and other stuff.

BenchmarkTools.Trial:
  memory estimate:  2.57 GiB
  allocs estimate:  56093582
  --------------
  minimum time:     4.790 s (14.77% GC)
  median time:      4.967 s (12.53% GC)
  mean time:        4.967 s (12.53% GC)
  maximum time:     5.144 s (10.44% GC)
  --------------
  samples:          2
  evals/sample:     1

# After triangle optimization.

BenchmarkTools.Trial:
  memory estimate:  1.41 GiB
  allocs estimate:  38067980
  --------------
  minimum time:     2.850 s (8.64% GC)
  median time:      2.958 s (12.03% GC)
  mean time:        2.958 s (12.03% GC)
  maximum time:     3.067 s (15.18% GC)
  --------------
  samples:          2
  evals/sample:     1

# After to_grid optimization.

BenchmarkTools.Trial:
  memory estimate:  1.40 GiB
  allocs estimate:  37743836
  --------------
  minimum time:     2.957 s (8.74% GC)
  median time:      2.967 s (13.21% GC)
  mean time:        2.967 s (13.21% GC)
  maximum time:     2.977 s (17.64% GC)
  --------------
  samples:          2
  evals/sample:     1

# TRIANGLES

## Initial

BenchmarkTools.Trial:
  memory estimate:  3.27 KiB
  allocs estimate:  51
  --------------
  minimum time:     5.122 μs (0.00% GC)
  median time:      5.322 μs (0.00% GC)
  mean time:        6.162 μs (5.28% GC)
  maximum time:     564.556 μs (97.99% GC)
  --------------
  samples:          10000
  evals/sample:     9

## Rewrote _to_ray_CS

BenchmarkTools.Trial:
  memory estimate:  1.91 KiB
  allocs estimate:  20
  --------------
  minimum time:     970.000 ns (0.00% GC)
  median time:      1.100 μs (0.00% GC)
  mean time:        1.398 μs (8.06% GC)
  maximum time:     291.540 μs (99.28% GC)
  --------------
  samples:          10000
  evals/sample:     10

## Rewrote t_scaled

BenchmarkTools.Trial:
  memory estimate:  1.78 KiB
  allocs estimate:  19
  --------------
  minimum time:     960.000 ns (0.00% GC)
  median time:      1.060 μs (0.00% GC)
  mean time:        1.320 μs (8.60% GC)
  maximum time:     294.820 μs (99.42% GC)
  --------------
  samples:          10000
  evals/sample:     10


## Put vertices on a stack instead of a heap

BenchmarkTools.Trial:
  memory estimate:  1.53 KiB
  allocs estimate:  17
  --------------
  minimum time:     738.400 ns (0.00% GC)
  median time:      850.400 ns (0.00% GC)
  mean time:        1.183 μs (9.43% GC)
  maximum time:     38.427 μs (95.92% GC)
  --------------
  samples:          10000
  evals/sample:     125

## Put all triangle attributes on a stack

BenchmarkTools.Trial:
  memory estimate:  1.17 KiB
  allocs estimate:  14
  --------------
  minimum time:     775.758 ns (0.00% GC)
  median time:      805.051 ns (0.00% GC)
  mean time:        927.754 ns (7.55% GC)
  maximum time:     30.790 μs (96.28% GC)
  --------------
  samples:          10000
  evals/sample:     99
