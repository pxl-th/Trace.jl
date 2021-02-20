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
