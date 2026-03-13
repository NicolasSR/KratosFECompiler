import sympy

import original_kratos_symbolic.kratos_symbolic_utilities as ksu

def main(options_dict):

    ## Symbolic generation settings
    dim = options_dict["dim"]

    # Define further options based on formulation
    if dim == 2:
        nnodes = 3
        strain_size = 3
    elif dim == 3:
        nnodes = 4
        strain_size = 6
    else:
        err_msg = "Incompatible number of dimensions: " + str(dim)
        raise Exception(err_msg)


    impose_partion_of_unity = False
    N,DN = ksu.DefineShapeFunctions(nnodes, dim, impose_partion_of_unity)

    #defining the unknowns
    v = ksu.DefineMatrix('v_nodes',nnodes,dim) #v(i,j) is velocity of node i component j
    vn = ksu.DefineMatrix('vn_nodes',nnodes,dim) #velocity one step back
    vnn = ksu.DefineMatrix('vnn_nodes',nnodes,dim) #velocity two step back
    p = ksu.DefineVector('p_nodes',nnodes)

    #define test functions
    w = ksu.DefineMatrix('w_nodes',nnodes,dim)
    q = ksu.DefineVector('q_nodes',nnodes)

    #define other nodally varying data
    f = ksu.DefineMatrix('f_nodes',nnodes,dim)

    #constitutive matrix
    C = ksu.DefineSymmetricMatrix('C_gauss',strain_size,strain_size)

    #define other symbols
    rho = sympy.Symbol('rho_gauss', positive=True)
    tau1 = sympy.Symbol('tau1_gauss', positive=True)
    tau2 = sympy.Symbol('tau2_gauss')

    bdf0 = sympy.Symbol('bdf0_gauss')
    bdf1 = sympy.Symbol('bdf1_gauss')
    bdf2 = sympy.Symbol('bdf2_gauss')

    #interpolaing to the gauss point
    fgauss = ksu.DefineVector('fg_gauss',dim)
    vgauss = ksu.DefineVector('vg_gauss',dim)
    acch = ksu.DefineVector('acchg_gauss',dim)
    pgauss = ksu.sympy.Symbol('pg_gauss')

    wgauss = w.transpose()*N
    qgauss = q.transpose()*N

    #computing gradients
    grad_v = DN.transpose()*v
    grad_q = DN.transpose()*q

    grad_p = ksu.DefineVector('grad_pg_gauss',dim)
    if(dim == 2):
        stress = ksu.DefineVector('stress_gauss',3)
    elif(dim == 3):
        stress = ksu.DefineVector('stress_gauss',6)

    #v_DV = vgauss.transpose()*grad_v
    B = ksu.MatrixB(DN)
    grad_sym_f = ksu.grad_sym_voigtform(DN,f)

    div_v = ksu.div(DN,v)
    div_w = ksu.div(DN,w)
    grad_sym_w = ksu.grad_sym_voigtform(DN,w)

    a = bdf0*v+bdf1*vn+bdf2*vnn
    grad_sym_res = ksu.grad_sym_voigtform(DN,f-rho*a)


    #compute galerkin functional
    rv_galerkin =  wgauss.transpose()*(fgauss - rho*acch) +  div_w*pgauss - grad_sym_w.transpose()*stress  -qgauss*div_v

    rv_stab = rho*div_w*tau2*div_v +  grad_q.transpose()*(rho*tau1*(fgauss - rho*acch - grad_p) )
    rv_stab += -tau1*grad_sym_w.transpose()*C*grad_sym_res ##TODO: it might be better to remove this term - it is the symmetric_gradient of the subscale!

    rv = rv_galerkin + rv_stab

    #define dofs & test function vector
    dofs = sympy.zeros(nnodes*(dim+1), 1)
    testfunc = sympy.zeros(nnodes*(dim+1), 1)
    for i in range(0,nnodes):
        for k in range(0,dim):
            dofs[i*(dim+1)+k] = v[i,k]
            testfunc[i*(dim+1)+k] = w[i,k]
        dofs[i*(dim+1)+dim] = p[i,0]
        testfunc[i*(dim+1)+dim] = q[i,0]
    print("dofs = ",dofs)

    rhs = ksu.Compute_RHS(rv, testfunc, do_simplifications = False)


    ##HERE WE MUST SUBSTITUTE EXPRESSIONS
    strain = ksu.grad_sym_voigtform(DN,v)
    rhs_to_derive = rhs.copy()

    #vector values
    rhs_to_derive = ksu.SubstituteMatrixValue( rhs_to_derive, fgauss, f.transpose()*N)
    rhs_to_derive = ksu.SubstituteMatrixValue( rhs_to_derive, vgauss, v.transpose()*N)
    rhs_to_derive = ksu.SubstituteMatrixValue( rhs_to_derive, acch, (bdf0*v+bdf1*vn+bdf2*vnn).transpose()*N )
    rhs_to_derive = ksu.SubstituteMatrixValue( rhs_to_derive, stress, C*strain)
    rhs_to_derive = ksu.SubstituteMatrixValue( rhs_to_derive, grad_p, DN.transpose()*p)

    #scalar values
    pgauss_expanded = (p.transpose()*N)[0]
    rhs_to_derive = ksu.SubstituteScalarValue( rhs_to_derive, pgauss,pgauss_expanded )

    ##obtain LHS by symbolic derivation
    lhs = ksu.Compute_LHS(rhs_to_derive, testfunc, dofs, do_simplifications = False)

    # rhs_out = ksu.OutputVector_CollectingFactors(rhs,"rRightHandSideVector",'c')
    # lhs_out = ksu.OutputMatrix_CollectingFactors(lhs,"rLeftHandSideMatrix",'c')

    w_g = sympy.Symbol("w_g", positive=True) # Gauss point integration weight

    print(f"Computing {dim}D{nnodes}N RHS Gauss point contribution\n")
    rhs_out = ksu.OutputVector_CollectingFactors(w_g*rhs, "rRightHandSideVector", "c", assignment_op='+=')

    print(f"Computing {dim}D{nnodes}N LHS Gauss point contribution\n")
    lhs_out = ksu.OutputMatrix_CollectingFactors(w_g*lhs, "rLeftHandSideMatrix", "c", assignment_op='+=')

    return rhs_out, lhs_out