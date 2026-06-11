import sympy
import original_kratos_symbolic.kratos_symbolic_utilities as ksu

def main(options_dict):

    ## Settings explanation
    # DIMENSION TO COMPUTE:
    # This symbolic generator is valid for both 2D and 3D cases. Since the element has been programed with a dimension template in Kratos,
    # it is advised to set the dim_to_compute flag as "Both". In this case the generated .cpp file will contain both 2D and 3D implementations.
    # LINEARISATION SETTINGS:
    # FullNR considers the convective velocity as "v-vmesh", hence v is taken into account in the derivation of the LHS and RHS.
    # Picard (a.k.a. QuasiNR) considers the convective velocity as "a", thus it is considered as a constant in the derivation of the LHS and RHS.
    # DIVIDE BY RHO:
    # If set to true, divides the mass conservation equation by rho in order to have a better conditioned matrix. Otherwise the original form is kept.
    # ARTIFICIAL COMPRESSIBILITY:
    # If set to true, the time derivative of the density is introduced in the mass conservation equation together with the state equation
    # dp/drho=c^2 (being c the sound velocity). Besides, the velocity divergence is not considered to be 0. These assumptions add some extra terms
    # to the usual Navier-Stokes equations that act as a weak compressibility controlled by the value of "c".
    # CONVECTIVE TERM:
    # If set to true, the convective term is taken into account in the calculation of the variational form. This allows generating both
    # Navier-Stokes and Stokes elements.

    ## Symbolic generation settings
    dim = options_dict["dim"]
    nnodes = options_dict["nnodes"]
    formulation = "Stokes"
    divide_by_rho = options_dict["divide_by_rho"]
    ASGS_stabilization = options_dict["ASGS_stabilization"]

    # Define further options based on formulation "Stokes"
    darcy_term = False
    convective_term = False
    artificial_compressibility = False

    info_msg = "\n"
    info_msg += "Element generator settings:\n"
    info_msg += "\t - Element type: " + formulation + "\n"
    info_msg += "\t - Dimensions: " + str(dim) + "\n"
    info_msg += "\t - Number of nodes: " + str(nnodes) + "\n"
    info_msg += "\t - ASGS stabilization: " + str(ASGS_stabilization) + "\n"
    info_msg += "\t - Pseudo-compressibility: " + str(artificial_compressibility) + "\n"
    info_msg += "\t - Divide mass conservation by rho: " + str(divide_by_rho) + "\n"
    print(info_msg)

    if dim == 2:
        strain_size = 3
    elif dim == 3:
        strain_size = 6

    N,DN = ksu.DefineShapeFunctions(nnodes, dim, impose_partion_of_unity = False)

    ## Unknown fields definition
    v = ksu.DefineMatrix('v_nodes',nnodes,dim)            # Current step velocity (v(i,j) refers to velocity of node i component j)
    vn = ksu.DefineMatrix('vn_nodes',nnodes,dim)          # Previous step velocity
    vnn = ksu.DefineMatrix('vnn_nodes',nnodes,dim)        # 2 previous step velocity
    p = ksu.DefineVector('p_nodes',nnodes)                # Pressure
    pn = ksu.DefineVector('pn_nodes',nnodes)              # Previous step pressure
    pnn = ksu.DefineVector('pnn_nodes',nnodes)            # 2 previous step pressure

    ## Fluid properties
    if artificial_compressibility:
        # If weak-compressibility is on, the density (rho) and speed of sound (c) become nodal variables
        rho_nodes = ksu.DefineVector('rho_nodes',nnodes) # Nodal density
        c_nodes = ksu.DefineVector('c_nodes',nnodes)     # Nodal sound speed
        rho = rho_nodes.transpose()*N          # Density Gauss pt. interpolation
        c = c_nodes.transpose()*N              # Sound speed Gauss pt. interpolation
        rho = rho[0]
        c = c[0]
    else:
        # With no weak-compressibility, the density (rho) is retrieved from the element properties and there is no speed of sound need
        rho = sympy.Symbol('rho_gauss', positive = True)     # Density

    ## Test functions definition
    w = ksu.DefineMatrix('w_nodes',nnodes,dim)            # Velocity field test function
    q = ksu.DefineVector('q_nodes',nnodes)                # Pressure field test function

    ## Other data definitions
    f = ksu.DefineMatrix('f_nodes',nnodes,dim)            # Forcing term
    v_sol_frac = ksu.DefineMatrix('v_sol_frac_nodes',nnodes,dim) # Solid fraction velocity

    ## Constitutive matrix definition
    C = ksu.DefineSymmetricMatrix('C_gauss',strain_size,strain_size)

    ## Stress vector definition
    stress = ksu.DefineVector('stress_gauss',strain_size)

    ## Other simbols definition
    dt = sympy.Symbol('dt_gauss', positive = True)         # Time increment
    mu = sympy.Symbol('mu_gauss', positive = True)         # Dynamic viscosity
    h = sympy.Symbol('h_gauss', positive = True)
    dyn_tau = sympy.Symbol('dyn_tau_gauss', positive = True)
    stab_c1 = sympy.Symbol('stab_c1_gauss', positive = True)
    stab_c2 = sympy.Symbol('stab_c2_gauss', positive = True)
    gauss_weight = sympy.Symbol('gauss_weight', positive = True)

    ## Symbols for Darcy term
    sigma = sympy.Symbol('sigma_gauss', positive = True) if darcy_term else 0.0 # Resistance (permeability inverse, 1/m^2)
    stab_c3 = sympy.Symbol('stab_c3_gauss', positive = True) if darcy_term else 0.0 # Darcy term stabilization constant

    ## Backward differences coefficients
    bdf0 = sympy.Symbol('bdf0_gauss')
    bdf1 = sympy.Symbol('bdf1_gauss')
    bdf2 = sympy.Symbol('bdf2_gauss')

    ## Data interpolation to the Gauss points
    f_gauss = f.transpose()*N
    v_gauss = v.transpose()*N
    v_sol_frac_gauss = v_sol_frac.transpose()*N

    ## Convective velocity definition
    if convective_term:
        if (linearisation == "Picard"):
            vconv = ksu.DefineMatrix('vconv_nodes',nnodes,dim)    # Convective velocity defined a symbol
        elif (linearisation == "FullNR"):
            vmesh = ksu.DefineMatrix('vmesh_nodes',nnodes,dim)    # Mesh velocity
            vconv = v - vmesh                           # Convective velocity defined as a velocity dependent variable
        else:
            raise Exception("Wrong linearisation \'" + linearisation + "\' selected. Available options are \'Picard\' and \'FullNR\'.")
        vconv_gauss = vconv.transpose()*N

    ## Compute the stabilization parameters
    if convective_term:
        stab_norm_a = 0.0
        for i in range(0, dim):
            stab_norm_a += vconv_gauss[i]**2
        stab_norm_a = sympy.sqrt(stab_norm_a)
        tau1 = 1.0/(rho*dyn_tau/dt + stab_c2*rho*stab_norm_a/h + stab_c1*mu/h**2 + stab_c3*sigma/h)    # Stabilization parameter 1
        tau2 = mu + (stab_c2*rho*stab_norm_a*h + stab_c3*sigma)/stab_c1                                # Stabilization parameter 2
    else:
        tau1 = 1.0/(rho*dyn_tau/dt + stab_c1*mu/h**2 + stab_c3*sigma/h)    # Stabilization parameter 1
        tau2 = mu + stab_c3*sigma/stab_c1                                  # Stabilization parameter 2

    ## Compute the rest of magnitudes at the Gauss points
    accel_gauss = (bdf0*v + bdf1*vn + bdf2*vnn).transpose()*N

    p_gauss = p.transpose()*N
    if artificial_compressibility:
        pder_gauss = (bdf0*p + bdf1*pn + bdf2*pnn).transpose()*N

    w_gauss = w.transpose()*N
    q_gauss = q.transpose()*N

    ## Gradients computation (fluid dynamics gradient)
    grad_w = ksu.DfjDxi(DN,w)
    grad_q = ksu.DfjDxi(DN,q)
    grad_p = ksu.DfjDxi(DN,p)
    grad_v = ksu.DfjDxi(DN,v)
    if artificial_compressibility:
        grad_rho = ksu.DfjDxi(DN,rho_nodes)

    div_w = ksu.div(DN,w)
    div_v = ksu.div(DN,v)
    if convective_term:
        div_vconv = ksu.div(DN,vconv)

    grad_sym_v = ksu.grad_sym_voigtform(DN,v)       # Symmetric gradient of v in Voigt notation
    grad_w_voigt = ksu.grad_sym_voigtform(DN,w)     # Symmetric gradient of w in Voigt notation
    # Recall that the grad(w):stress contraction equals grad_sym(w)*stress in Voigt notation since the stress is a symmetric tensor.

    # Convective term definition
    if convective_term:
        convective_term_gauss = (vconv_gauss.transpose()*grad_v)
        rho_convective_term_gauss = vconv_gauss.transpose()*grad_rho

    ## Compute galerkin functional
    # Navier-Stokes functional
    if divide_by_rho:
        rv_galerkin = rho*w_gauss.transpose()*f_gauss - rho*w_gauss.transpose()*accel_gauss - grad_w_voigt.transpose()*stress + div_w*p_gauss - sigma*w_gauss.transpose()*(v_gauss-v_sol_frac_gauss) - q_gauss*div_v
        if artificial_compressibility:
            rv_galerkin -= (1/(rho*c*c))*q_gauss*pder_gauss
            if convective_term:
                rv_galerkin -= (1/rho)*q_gauss*rho_convective_term_gauss
        if convective_term:
            rv_galerkin -= rho*w_gauss.transpose()*convective_term_gauss.transpose()
    else:
        rv_galerkin = rho*w_gauss.transpose()*f_gauss - rho*w_gauss.transpose()*accel_gauss  - grad_w_voigt.transpose()*stress + div_w*p_gauss - rho*q_gauss*div_v
        if artificial_compressibility:
            rv_galerkin -= (1/(c*c))*q_gauss*pder_gauss
            if convective_term:
                rv_galerkin -= q_gauss*rho_convective_term_gauss
        if convective_term:
            rv_galerkin -= rho*w_gauss.transpose()*convective_term_gauss.transpose()

    ##  Stabilization functional terms
    # Momentum conservation residual
    # Note that the viscous stress term is dropped since linear elements are used
    vel_residual = rho*f_gauss - rho*accel_gauss - grad_p - sigma*(v_gauss-v_sol_frac_gauss)
    if convective_term:
        vel_residual -= rho*convective_term_gauss.transpose()

    # Mass conservation residual
    if divide_by_rho:
        mas_residual = -div_v
        if artificial_compressibility:
            mas_residual -= (1/(rho*c*c))*pder_gauss
            if convective_term:
                mas_residual -= (1/rho)*rho_convective_term_gauss
    else:
        mas_residual = -rho*div_v
        if artificial_compressibility:
            mas_residual -= (1/(c*c))*pder_gauss
            if convective_term:
                mas_residual -= rho_convective_term_gauss

    vel_subscale = tau1*vel_residual
    mas_subscale = tau2*mas_residual

    # Compute the ASGS stabilization terms using the momentum and mass conservation residuals above
    if divide_by_rho:
        rv_stab = grad_q.transpose()*vel_subscale
    else:
        rv_stab = rho*grad_q.transpose()*vel_subscale
    if convective_term:
        rv_stab += rho*vconv_gauss.transpose()*grad_w*vel_subscale
        rv_stab += rho*div_vconv*w_gauss.transpose()*vel_subscale
    rv_stab -= sigma*w_gauss.transpose()*vel_subscale
    rv_stab += div_w*mas_subscale

    ## Add the stabilization terms to the original residual terms
    if ASGS_stabilization:
        rv = rv_galerkin + rv_stab
    else:
        rv = rv_galerkin


    ## Define DOFs and test function vectors
    dofs = sympy.zeros(nnodes*(dim+1), 1)
    testfunc = sympy.zeros(nnodes*(dim+1), 1)

    for i in range(nnodes):

        # Velocity DOFs and test functions
        for k in range(0,dim):
            dofs[i*(dim+1)+k] = v[i,k]
            testfunc[i*(dim+1)+k] = w[i,k]

        # Pressure DOFs and test functions
        dofs[i*(dim+1)+dim] = p[i,0]
        testfunc[i*(dim+1)+dim] = q[i,0]

    ## Compute LHS and RHS

    w_g = sympy.Symbol("w_g", positive=True) # Gauss point integration weight

    # For the RHS computation one wants the residual of the previous iteration (residual based formulation). By this reason the stress is
    # included as a symbolic variable, which is assumed to be passed as an argument from the previous iteration database.
    print(f"Computing {dim}D{nnodes}N RHS Gauss point contribution\n")
    rhs = ksu.Compute_RHS(rv.copy(), testfunc, do_simplifications = False)
    rhs_out = ksu.OutputVector_CollectingFactors(w_g*rhs, "rRightHandSideVector", "c", assignment_op='+=')


    # Compute LHS (RHS(residual) differenctiation w.r.t. the DOFs)
    # Note that the 'stress' (symbolic variable) is substituted by 'C*grad_sym_v' for the LHS differenctiation. Otherwise the velocity terms
    # within the velocity symmetryc gradient would not be considered in the differenctiation, meaning that the stress would be considered as
    # a velocity independent constant in the LHS.
    print(f"Computing {dim}D{nnodes}N LHS Gauss point contribution\n")
    ksu.SubstituteMatrixValue(rhs, stress, C*grad_sym_v)
    lhs = ksu.Compute_LHS(rhs, testfunc, dofs, do_simplifications = False) # Compute the LHS (considering stress as C*(B*v) to derive w.r.t. v)
    lhs_out = ksu.OutputMatrix_CollectingFactors(w_g*lhs, "rLeftHandSideMatrix", "c", assignment_op='+=')
    
    return rhs_out, lhs_out