#=
    QOptics
    Copyright (c) 2018 Ahmed Omran <aomran@fas.harvard.edu>

    Distributed under terms of the MIT license.
=#

module RydbergArrays

    using Combinatorics: combinations
    import QuantumOptics
    qo = QuantumOptics
    using SparseArrays

    export createAtomArray, createOperators, createProjectors
    export createSpinOperators, VdWInteraction, createVdWInteractionTerm


    """
        createAtomArray(N)

    Generate atom array of length `N`, where all atoms are in the ground state
    
    #  Arguments
    - `N::Integer`: Number of atoms
    """
    function createAtomArray(N::Int)
        spin_1_2 = qo.SpinBasis(1//2)
        ψ₀ = qo.Ket(spin_1_2, [0,1])
        Atoms = [ψ₀ for i in 1:N]
        return qo.tensor(Atoms...)
    end


    """
        createOperators(N::Int, op::QuantumOptics.operator.Operator)

    Generate set of single-particle operators operating on a many-body system.
    Tensor product of operator is taken with (N-1) identity operators. This
    seems to be slightly more efficient than the native 'embed' operation of
    QuantumOptics.jl.

    # Arguments
    - `N::Integer`: Number of atoms
    - `op::QuantumOptics.operator.Operator`: Single-particle operator
    """
    function createOperators(N::Int, op::qo.AbstractOperator)
        spin_1_2 = qo.SpinBasis(1//2)
        Identity = qo.identityoperator(spin_1_2)
        opList = [[op];repeat([Identity],N-1)]
        Operators = [qo.tensor(circshift(opList,i)...) for i in 0:N-1]
	return Operators
    end

    """
        createSpinOperators(N::Int)

    Generate projectors onto ground and Rydberg states
    
    # Arguments
    - `N::Integer`: Number of atoms
    """
    function createProjectors(N::Int)
        spin_1_2 = qo.SpinBasis(1//2)
        ψ₀ = qo.Ket(spin_1_2, [0,1])
        ψᵣ = qo.Ket(spin_1_2, [1,0])
        P₀ = createOperators(N, qo.projector(ψ₀))
        Pᵣ = createOperators(N, qo.projector(ψᵣ))
        return P₀, Pᵣ
    end


    """
        createSpinOperators(N::Int)

    Generate set of spin-operators on a many-body system.
    
    # Arguments
    - `N::Integer`: Number of atoms
    """
    function createSpinOperators(N::Int)
        spin_1_2 = qo.SpinBasis(1//2)
        σˣ = createOperators(N, qo.sigmax(spin_1_2))
        σʸ = createOperators(N, qo.sigmay(spin_1_2))
        σᶻ = createOperators(N, qo.sigmaz(spin_1_2))
        σ⁺ = createOperators(N, qo.sigmap(spin_1_2))
        σ⁻ = createOperators(N, qo.sigmam(spin_1_2))

        return σˣ, σʸ, σᶻ, σ⁺, σ⁻
    end


    """
        VdWInteraction(x::Float64, y::Float64, C₆::Float64)

    Calculate interaction term between two Rydberg atoms.

    # Arguments
    - `x::Float64`: Position of first atom
    - `y::Float64`: Position of second atom
    - `C₆::Float64`: van der Waals coefficient
    """
    function VdWInteraction(x::Float64,y::Float64,C₆::Float64)
        return C₆/abs(x-y)^6
    end


    """
        createVdWInteractionTerm(N::Int, C₆::Float64, d::Float64)

    Generate Hamiltonian term of Rydberg interactions.
    
    # Arguments
    - `N::Integer`: Number of atoms
    - `C₆::Float64`: van der Waals coefficient
    - `d::Float64`: Spacing between neighbouring pairs of atoms
    """
    function createVdWInteractionTerm(N::Int, C₆::Float64, d::Float64)
        spin_1_2 = qo.SpinBasis(1//2)
        ψᵣ = qo.Ket(spin_1_2, [1,0])
        Pᵣ = createOperators(N, qo.projector(ψᵣ))
        Interactions = [VdWInteraction(i[1]*d, i[2]*d, C₆) for i in combinations(1:N, 2)]
        Projectors = [Pᵣ[i[1]]*Pᵣ[i[2]] for i in combinations(1:N, 2)]
        return sum(Interactions.*Projectors)
    end


    """
        createVdWInteractionTerm(N::Int, C₆::Float64, d₁::Float64, d₂::Float64)

    Generate Hamiltonian term of Rydberg interactions, including a staggered spacing between atoms.

    # Arguments
    - `N::Integer`: Number of atoms
    - `C₆::Float64`: van der Waals coefficient
    - `d₁::Float64`: Even spacing between neighbouring pairs of atoms
    - `d₂::Float64`: Odd spacing between neighbouring pairs of atoms
    """
    function createVdWInteractionTerm(N::Int, C₆::Float64, d₁::Float64, d₂::Float64)
        spin_1_2 = qo.SpinBasis(1//2)
        ψᵣ = qo.Ket(spin_1_2, [1,0])
        Pᵣ = createOperators(N, qo.projector(ψᵣ))
        Interactions = [VdWInteraction(i[1]*d₁ + ((i[1]-1)÷2)*(d₂-d₁), i[2]*d₁ + ((i[2]-1)÷2)*(d₂-d₁), C₆) for i in combinations(1:N, 2)]
        Projectors = [Pᵣ[i[1]]*Pᵣ[i[2]] for i in combinations(1:N, 2)]
        return sum(Interactions.*Projectors)
    end


end
