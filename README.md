# Trace.jl

Ray tracing.

## Install

Right now it is Windows only due to custom version of `Assimp` library.
Hovewer this will be fixed.

## Examples

All of the examples are available in `scenes/` directory.

Example of running caustic glass scene (note that it supports multithreading):

```bash
julia -t 4 --project=. scenes/caustic_glass.jl
```

### Caustic simulation

Caustic effect using Stochastic Progressive Photon Mapping on a glass block.

|Image|Video|
|:-:|:-:|
|<img src="scenes/caustic-glass-sppm-100-iterations.png" width="400">|<img src="https://i.ytimg.com/vi_webp/87NlMA3Vwvs/maxresdefault.webp" width="720">|
|100 iterations, ray depth 8|25 iterations per frame, ray depth 5. [YouTube](https://www.youtube.com/watch?v=87NlMA3Vwvs)|
