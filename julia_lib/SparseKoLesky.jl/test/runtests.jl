using SparseKoLesky: SparseKoLesky as KL
using Test: @testset, @test

@testset "SparseKoLesky.jl" begin
    # Write your tests here.
    @testset "MaximinNN.jl" begin
        include("test_MaximinNN.jl")
    end
end
