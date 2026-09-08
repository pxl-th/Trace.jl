# DELETED in phase 1.5: see Mantle/docs/mantle-owns-it.md
#
# 66 lines of generated precompile statements naming `LavaArray`,
# `LavaBackend`, `LavaDevice`, `MVE` and `VulkanInstanceRecord` — 138 backend
# references in a package that must name none. Regenerating them against
# whichever backend is present is phase 2.8's, if they are worth having at all.

# A no-op, so the workload above still has something to call.
#
# The 138 generated statements this file held named `LavaArray`, `LavaBackend`,
# `LavaDevice`, `MVE` and `VulkanInstanceRecord` — a package that must name no
# backend, listing five of one backend's types 138 times. They bought ~22 s of
# inference per session in ten signatures, which is worth having back; what they
# cannot be is generated against one backend and checked in.
#
# Regenerating them against whichever backend is present is phase 2.9's, and
# `precompile` on a signature is device-free by construction, so there is
# nothing here that has to name a driver.
_precompile_statements() = nothing
