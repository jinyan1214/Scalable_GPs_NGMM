# Kernel Functions
#------------------------------
using LinearAlgebra
using StaticArrays
using KernelFunctions
using FastGaussQuadrature: gausslobatto
using Functors

export PathKernel, HybridDSKernel