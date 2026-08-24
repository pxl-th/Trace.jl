using Test
using Hikari

# `src/precompile_statements.jl` is generated from a --trace-compile-timing run
# and names concrete types like
# `Scene{TLAS{LavaBackend}, MultiTypeSet{LavaBackend}, ...}` in full. Rename a
# type parameter, reorder a field, change a keyword and the affected signature
# stops matching any method -- `precompile` then returns `false` and simply
# caches nothing. Startup gets slower and no test fails, which is exactly the
# kind of rot that goes unnoticed for months.
#
# So assert every statement still resolves. When this fails, regenerate the file
# rather than deleting the offending line: a signature that no longer matches
# usually means the real call site moved too.
@testset "precompile statements all resolve" begin
    ok, total = Hikari._precompile_statements()
    @test total > 0
    @test ok == total
    if ok != total
        @info "stale precompile signatures" resolved = ok attempted = total
    end
end
