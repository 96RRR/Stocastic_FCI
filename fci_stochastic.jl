using LinearAlgebra

function ϕI_H_ϕJ(DI, DJ, hmo, Mmo)
    nmo = size(DI, 1)
    gIJ, det_S = construct_g(DI, DJ)

    first = 2.0 * det_S * pyscf.lib.einsum("rs,rs->", hmo, gIJ)
    second = 0.00
    second = pyscf.lib.einsum("pqrs,pq,rs->", Mmo, gIJ, gIJ)
    return first + second, det_S
end


function construct_D(Jmax, nmo, no, Orb_en)
    # D_J_p_i
    D = zeros((Jmax, nmo, no))
    for p = 1:nmo, i = 1:no
        if (p == i)
            D[1, p, i] = 1
        end
    end
    for J = 2:Jmax
        for i = 1:no
            for p = 1:nmo
                if p <= no
                   D[J, p, i] = rand((-1,1)) 
                elseif p > no
                   D[J, p, i] = rand((-1,1)) * exp(-0.1*(Orb_en[p]/Orb_en[no+1]-1))
                end
            end
            norm = D[J, :, i]' * D[J, :, i]
            D[J,:,i] = D[J,:,i]/sqrt(norm)
        end
        #display(D[J,:,:])
    end
    
    return D
end

function construct_ΔIJ(DI, DJ)
    nmo = size(DI, 1)
    no = size(DI, 2)
    Δ = zeros(no, no)
    for i = 1:no, j = 1:no, p = 1:nmo
        Δ[i, j] += DI[p,i] * DJ[p,j]
    end
    return Δ
end

function construct_g(DI, DJ)
    nmo = size(DI, 1)
    no = size(DI, 2)
    Δ = construct_ΔIJ(DJ, DI)
    U, S, V = svd(Δ)
    det_S = 1.0
    for j = 1:no
        det_S *= S[j]
    end
    tmp_p = ones(length(S))
    for j = 1:no, i = 1:no
        if j != i
            tmp_p[j] *= S[i]
        end
    end
    p = Diagonal(tmp_p)
    tmp_gIJ = (DI * V * p * U' * DJ')
    gIJ = tmp_gIJ'
    return gIJ, det_S
end

function get_energy(Jmax, Orb_en)
    D = construct_D(Jmax, mol.nao, mol.nelectron ÷ 2, Orb_en)

    H = zeros(Jmax, Jmax)
    S = zeros(Jmax, Jmax)
    for I = 1:Jmax, J = 1:Jmax
        DI = D[I, :, :]
        DJ = D[J, :, :]
        H[I,J], det_S = ϕI_H_ϕJ(DI, DJ, hmo, Mmo)
        S[I,J] = det_S^2
    end
    H = Symmetric(H)
    S = Symmetric(S)
    e,v = eigen(H,S)
    #display(e)
    eigen(S)
    e .+= mol.energy_nuc()
    return e[1:4]
end


using PyCall
pyscf = pyimport("pyscf")

#mol = pyscf.gto.M(atom="O 0.00 0.00 -0.004762593; H 0.00 0.801842329 -0.560344467; H 0.00 -0.801842329 -0.560344467; ", basis="cc-pvdz")
mol = pyscf.gto.M(atom="H 0.00 0.00 0.00; H 0.00 0.00 0.714; H 0.714 0.00 0.00; H 0.714 0.00 0.714; ", basis="cc-pvdz")
rhf = mol.RHF()
rhf.run()
C = rhf.mo_coeff
Orb_en = rhf.mo_energy

hao = mol.intor("int1e_nuc") + mol.intor("int1e_kin");
hmo = C' * hao * C;

gao = mol.intor("int2e");
gmo = pyscf.ao2mo.incore.full(gao, C);

Mmo = 2*gmo - permutedims(gmo, (1,4,3,2));

cisolver = pyscf.fci.FCI(rhf)
FCI_energy = cisolver.kernel()[1]
@show rhf.e_tot FCI_energy;


[get_energy(200, Orb_en) for _ = 1:10] |> display
