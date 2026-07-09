#include "custom_curl_condition.h"

namespace Kratos
{

///@name Specialized implementation of VMS for functions that depend on TDim
///@{

/**
 * @see CustomCurlCondition::EquationIdVector
 */
template <>
void CustomCurlCondition<2,2>::EquationIdVector(
    EquationIdVectorType& rResult,
    const ProcessInfo& rCurrentProcessInfo) const
{
    const unsigned int NumNodes = 2;
    const unsigned int LocalSize = 6;
    unsigned int LocalIndex = 0;

    if (rResult.size() != LocalSize)
        rResult.resize(LocalSize, false);

    for (unsigned int iNode = 0; iNode < NumNodes; ++iNode)
    {
        rResult[LocalIndex++] = this->GetGeometry()[iNode].GetDof(VELOCITY_X).EquationId();
        rResult[LocalIndex++] = this->GetGeometry()[iNode].GetDof(VELOCITY_Y).EquationId();
        rResult[LocalIndex++] = this->GetGeometry()[iNode].GetDof(PRESSURE).EquationId();
    }
}

/**
 * @see CustomCurlCondition::EquationIdVector
 */
template <>
void CustomCurlCondition<3,3>::EquationIdVector(
    EquationIdVectorType& rResult,
    const ProcessInfo& rCurrentProcessInfo) const
{
    const SizeType NumNodes = 3;
    const SizeType LocalSize = 12;
    unsigned int LocalIndex = 0;

    if (rResult.size() != LocalSize)
        rResult.resize(LocalSize, false);

    for (unsigned int iNode = 0; iNode < NumNodes; ++iNode)
    {
        rResult[LocalIndex++] = this->GetGeometry()[iNode].GetDof(VELOCITY_X).EquationId();
        rResult[LocalIndex++] = this->GetGeometry()[iNode].GetDof(VELOCITY_Y).EquationId();
        rResult[LocalIndex++] = this->GetGeometry()[iNode].GetDof(VELOCITY_Z).EquationId();
        rResult[LocalIndex++] = this->GetGeometry()[iNode].GetDof(PRESSURE).EquationId();
    }
}

/**
 * @see CustomCurlCondition::GetDofList
 */
template <>
void CustomCurlCondition<2,2>::GetDofList(
    DofsVectorType& rElementalDofList,
    const ProcessInfo& rCurrentProcessInfo) const
{
    const SizeType NumNodes = 2;
    const SizeType LocalSize = 6;

    if (rElementalDofList.size() != LocalSize)
        rElementalDofList.resize(LocalSize);

    unsigned int LocalIndex = 0;

    for (unsigned int iNode = 0; iNode < NumNodes; ++iNode)
    {
        rElementalDofList[LocalIndex++] = this->GetGeometry()[iNode].pGetDof(VELOCITY_X);
        rElementalDofList[LocalIndex++] = this->GetGeometry()[iNode].pGetDof(VELOCITY_Y);
        rElementalDofList[LocalIndex++] = this->GetGeometry()[iNode].pGetDof(PRESSURE);
    }
}

/**
 * @see CustomCurlCondition::GetDofList
 */
template <>
void CustomCurlCondition<3,3>::GetDofList(
    DofsVectorType& rElementalDofList,
    const ProcessInfo& rCurrentProcessInfo) const
{
    const SizeType NumNodes = 3;
    const SizeType LocalSize = 12;

    if (rElementalDofList.size() != LocalSize)
        rElementalDofList.resize(LocalSize);

    unsigned int LocalIndex = 0;

    for (unsigned int iNode = 0; iNode < NumNodes; ++iNode)
    {
        rElementalDofList[LocalIndex++] = this->GetGeometry()[iNode].pGetDof(VELOCITY_X);
        rElementalDofList[LocalIndex++] = this->GetGeometry()[iNode].pGetDof(VELOCITY_Y);
        rElementalDofList[LocalIndex++] = this->GetGeometry()[iNode].pGetDof(VELOCITY_Z);
        rElementalDofList[LocalIndex++] = this->GetGeometry()[iNode].pGetDof(PRESSURE);
    }
}

template<>
void CustomCurlCondition<2,2>::ApplyNeumannCondition(
    MatrixType &rLeftHandSideMatrix,
    VectorType &rRightHandSideVector,
    const ProcessInfo& rCurrentProcessInfo
)
{

    KRATOS_INFO("CustomCurlCondition") << "Inside ApplyNeumannCondition" << std::endl;
    const unsigned int TDim = 2;
    const unsigned int TNumNodes = 2;

    const unsigned int LocalSize = TDim+1;
    const GeometryType& rGeom = this->GetGeometry();
    const GeometryData::IntegrationMethod integration_method = static_cast<GeometryData::IntegrationMethod> (GeometryData::IntegrationMethod::GI_GAUSS_2);
    const GeometryType::IntegrationPointsArrayType& IntegrationPoints = rGeom.IntegrationPoints(integration_method);
    const unsigned int NumGauss = IntegrationPoints.size();

    // Calculate Jacobians at integration points
    Matrix J;
    Matrix inv_Jt_J;
    double det_Jt_J;
    Vector det_Jt_J_vect(NumGauss);
    std::vector<BoundedMatrix<double, TDim, TDim-1>> J_pseudo_inv_vect(NumGauss);
    for (IndexType g = 0; g < NumGauss; ++g) {
        rGeom.Jacobian(J, g, integration_method); //J becomes of size TDim*(TDim-1)
        Matrix Jt_J = prod(trans(J), J);
        MathUtils<double>::InvertMatrix(Jt_J, inv_Jt_J, det_Jt_J);
        J_pseudo_inv_vect[g] = prod(inv_Jt_J,trans(J));
        det_Jt_J_vect[g] = det_Jt_J;
    }

    MatrixType NContainer = rGeom.ShapeFunctionsValues(integration_method);

    std::vector<BoundedMatrix<double, TNumNodes, TDim>> DNContainer(NumGauss);
    const auto& DN_De_flat = rGeom.ShapeFunctionsLocalGradients(integration_method);
    Matrix DN_De;
    for (IndexType g = 0; g < NumGauss; ++g) {
        DN_De = ZeroMatrix(TDim,TDim);
        DN_De(0,0) = DN_De_flat[g](0,0);
        DN_De(0,1) = DN_De_flat[g](1,0);
        DNContainer[g] = trans(prod(trans(J_pseudo_inv_vect[g]), DN_De));
    }

    for (unsigned int g = 0; g < NumGauss; g++)
    {
        Vector N = row(NContainer,g);
        BoundedMatrix<double, TNumNodes, TDim> DN = DNContainer[g];
        double w_g = sqrt(det_Jt_J_vect[g]) * IntegrationPoints[g].Weight();
        
        // Compute Unit normal
        array_1d<double, 3> n_gauss;
        n_gauss = rGeom.Normal(IntegrationPoints[g].Coordinates());
        double A = norm_2(n_gauss);
        n_gauss /= A;

        // Set nodal data
        array_1d<double, TNumNodes> nu_nodes;
        BoundedMatrix<double, TNumNodes, TDim> u_nodes;
        for (IndexType i = 0; i < TNumNodes; ++i) {
            nu_nodes[i] = rGeom[i].FastGetSolutionStepValue(DYNAMIC_VISCOSITY);
            const auto& r_v = rGeom[i].FastGetSolutionStepValue(VELOCITY);
            for (IndexType d = 0; d < TDim; ++d) {
                u_nodes(i, d) = r_v[d];
            }
        }

        const double delta_gauss = rCurrentProcessInfo.GetValue(TAUONE);
        
                const double crRightHandSideVector0 = delta_gauss*w_g*(N[0]*nu_nodes[0] + N[1]*nu_nodes[1] + N[2]*nu_nodes[2])*(DN(0,0)*u_nodes(0,1) - DN(0,1)*u_nodes(0,0) + DN(1,0)*u_nodes(1,1) - DN(1,1)*u_nodes(1,0) + DN(2,0)*u_nodes(2,1) - DN(2,1)*u_nodes(2,0));
        rRightHandSideVector[0]+=0;
        rRightHandSideVector[1]+=0;
        rRightHandSideVector[2]+=crRightHandSideVector0*(DN(0,0)*n_gauss[1] - DN(0,1)*n_gauss[0]);
        rRightHandSideVector[3]+=0;
        rRightHandSideVector[4]+=0;
        rRightHandSideVector[5]+=crRightHandSideVector0*(DN(1,0)*n_gauss[1] - DN(1,1)*n_gauss[0]);
        rRightHandSideVector[6]+=0;
        rRightHandSideVector[7]+=0;
        rRightHandSideVector[8]+=crRightHandSideVector0*(DN(2,0)*n_gauss[1] - DN(2,1)*n_gauss[0]);
        

                const double crLeftHandSideMatrix0 = delta_gauss*w_g*(N[0]*nu_nodes[0] + N[1]*nu_nodes[1] + N[2]*nu_nodes[2]);
        const double crLeftHandSideMatrix1 = crLeftHandSideMatrix0*(DN(0,0)*n_gauss[1] - DN(0,1)*n_gauss[0]);
        const double crLeftHandSideMatrix2 = crLeftHandSideMatrix0*(DN(1,0)*n_gauss[1] - DN(1,1)*n_gauss[0]);
        const double crLeftHandSideMatrix3 = crLeftHandSideMatrix0*(DN(2,0)*n_gauss[1] - DN(2,1)*n_gauss[0]);
        rLeftHandSideMatrix(0,0)+=0;
        rLeftHandSideMatrix(0,1)+=0;
        rLeftHandSideMatrix(0,2)+=0;
        rLeftHandSideMatrix(0,3)+=0;
        rLeftHandSideMatrix(0,4)+=0;
        rLeftHandSideMatrix(0,5)+=0;
        rLeftHandSideMatrix(0,6)+=0;
        rLeftHandSideMatrix(0,7)+=0;
        rLeftHandSideMatrix(0,8)+=0;
        rLeftHandSideMatrix(1,0)+=0;
        rLeftHandSideMatrix(1,1)+=0;
        rLeftHandSideMatrix(1,2)+=0;
        rLeftHandSideMatrix(1,3)+=0;
        rLeftHandSideMatrix(1,4)+=0;
        rLeftHandSideMatrix(1,5)+=0;
        rLeftHandSideMatrix(1,6)+=0;
        rLeftHandSideMatrix(1,7)+=0;
        rLeftHandSideMatrix(1,8)+=0;
        rLeftHandSideMatrix(2,0)+=DN(0,1)*crLeftHandSideMatrix1;
        rLeftHandSideMatrix(2,1)+=-DN(0,0)*crLeftHandSideMatrix1;
        rLeftHandSideMatrix(2,2)+=0;
        rLeftHandSideMatrix(2,3)+=DN(1,1)*crLeftHandSideMatrix1;
        rLeftHandSideMatrix(2,4)+=-DN(1,0)*crLeftHandSideMatrix1;
        rLeftHandSideMatrix(2,5)+=0;
        rLeftHandSideMatrix(2,6)+=DN(2,1)*crLeftHandSideMatrix1;
        rLeftHandSideMatrix(2,7)+=-DN(2,0)*crLeftHandSideMatrix1;
        rLeftHandSideMatrix(2,8)+=0;
        rLeftHandSideMatrix(3,0)+=0;
        rLeftHandSideMatrix(3,1)+=0;
        rLeftHandSideMatrix(3,2)+=0;
        rLeftHandSideMatrix(3,3)+=0;
        rLeftHandSideMatrix(3,4)+=0;
        rLeftHandSideMatrix(3,5)+=0;
        rLeftHandSideMatrix(3,6)+=0;
        rLeftHandSideMatrix(3,7)+=0;
        rLeftHandSideMatrix(3,8)+=0;
        rLeftHandSideMatrix(4,0)+=0;
        rLeftHandSideMatrix(4,1)+=0;
        rLeftHandSideMatrix(4,2)+=0;
        rLeftHandSideMatrix(4,3)+=0;
        rLeftHandSideMatrix(4,4)+=0;
        rLeftHandSideMatrix(4,5)+=0;
        rLeftHandSideMatrix(4,6)+=0;
        rLeftHandSideMatrix(4,7)+=0;
        rLeftHandSideMatrix(4,8)+=0;
        rLeftHandSideMatrix(5,0)+=DN(0,1)*crLeftHandSideMatrix2;
        rLeftHandSideMatrix(5,1)+=-DN(0,0)*crLeftHandSideMatrix2;
        rLeftHandSideMatrix(5,2)+=0;
        rLeftHandSideMatrix(5,3)+=DN(1,1)*crLeftHandSideMatrix2;
        rLeftHandSideMatrix(5,4)+=-DN(1,0)*crLeftHandSideMatrix2;
        rLeftHandSideMatrix(5,5)+=0;
        rLeftHandSideMatrix(5,6)+=DN(2,1)*crLeftHandSideMatrix2;
        rLeftHandSideMatrix(5,7)+=-DN(2,0)*crLeftHandSideMatrix2;
        rLeftHandSideMatrix(5,8)+=0;
        rLeftHandSideMatrix(6,0)+=0;
        rLeftHandSideMatrix(6,1)+=0;
        rLeftHandSideMatrix(6,2)+=0;
        rLeftHandSideMatrix(6,3)+=0;
        rLeftHandSideMatrix(6,4)+=0;
        rLeftHandSideMatrix(6,5)+=0;
        rLeftHandSideMatrix(6,6)+=0;
        rLeftHandSideMatrix(6,7)+=0;
        rLeftHandSideMatrix(6,8)+=0;
        rLeftHandSideMatrix(7,0)+=0;
        rLeftHandSideMatrix(7,1)+=0;
        rLeftHandSideMatrix(7,2)+=0;
        rLeftHandSideMatrix(7,3)+=0;
        rLeftHandSideMatrix(7,4)+=0;
        rLeftHandSideMatrix(7,5)+=0;
        rLeftHandSideMatrix(7,6)+=0;
        rLeftHandSideMatrix(7,7)+=0;
        rLeftHandSideMatrix(7,8)+=0;
        rLeftHandSideMatrix(8,0)+=DN(0,1)*crLeftHandSideMatrix3;
        rLeftHandSideMatrix(8,1)+=-DN(0,0)*crLeftHandSideMatrix3;
        rLeftHandSideMatrix(8,2)+=0;
        rLeftHandSideMatrix(8,3)+=DN(1,1)*crLeftHandSideMatrix3;
        rLeftHandSideMatrix(8,4)+=-DN(1,0)*crLeftHandSideMatrix3;
        rLeftHandSideMatrix(8,5)+=0;
        rLeftHandSideMatrix(8,6)+=DN(2,1)*crLeftHandSideMatrix3;
        rLeftHandSideMatrix(8,7)+=-DN(2,0)*crLeftHandSideMatrix3;
        rLeftHandSideMatrix(8,8)+=0;
        

    }
}

template<>
void CustomCurlCondition<3,3>::ApplyNeumannCondition(
    MatrixType &rLeftHandSideMatrix,
    VectorType &rRightHandSideVector,
    const ProcessInfo& rCurrentProcessInfo
)
{
    const unsigned int TDim = 3;
    const unsigned int TNumNodes = 3;

    const unsigned int LocalSize = TDim+1;
    const GeometryType& rGeom = this->GetGeometry();
    const GeometryData::IntegrationMethod integration_method = static_cast<GeometryData::IntegrationMethod> (GeometryData::IntegrationMethod::GI_GAUSS_2);
    const GeometryType::IntegrationPointsArrayType& IntegrationPoints = rGeom.IntegrationPoints(integration_method);
    const unsigned int NumGauss = IntegrationPoints.size();

    // Calculate Jacobians at integration points
    Matrix J;
    Matrix inv_Jt_J;
    double det_Jt_J;
    Vector det_Jt_J_vect(NumGauss);
    std::vector<BoundedMatrix<double, TDim, TDim-1>> J_pseudo_inv_vect(NumGauss);
    for (IndexType g = 0; g < NumGauss; ++g) {
        rGeom.Jacobian(J, g, integration_method); //J becomes of size TDim*(TDim-1)
        Matrix Jt_J = prod(trans(J), J);
        MathUtils<double>::InvertMatrix(Jt_J, inv_Jt_J, det_Jt_J);
        J_pseudo_inv_vect[g] = prod(inv_Jt_J,trans(J));
        det_Jt_J_vect[g] = det_Jt_J;
    }

    MatrixType NContainer = rGeom.ShapeFunctionsValues(integration_method);

    std::vector<BoundedMatrix<double, TNumNodes, TDim>> DNContainer(NumGauss);
    const auto& DN_De_flat = rGeom.ShapeFunctionsLocalGradients(integration_method);
    Matrix DN_De;
    for (IndexType g = 0; g < NumGauss; ++g) {
        DN_De = ZeroMatrix(TDim,TDim);
        DN_De(0,0) = DN_De_flat[g](0,0);
        DN_De(0,1) = DN_De_flat[g](1,0);
        DN_De(0,2) = DN_De_flat[g](2,0);
        DN_De(1,0) = DN_De_flat[g](0,1);
        DN_De(1,1) = DN_De_flat[g](1,1);
        DN_De(1,2) = DN_De_flat[g](2,1);
        DNContainer[g] = trans(prod(trans(J_pseudo_inv_vect[g]), DN_De));
    }

    for (unsigned int g = 0; g < NumGauss; g++)
    {
        Vector N = row(NContainer,g);
        BoundedMatrix<double, TNumNodes, TDim> DN = DNContainer[g];
        double w_g = sqrt(det_Jt_J_vect[g]) * IntegrationPoints[g].Weight();

        // Compute Unit normal
        array_1d<double, 3> n_gauss;
        n_gauss = rGeom.Normal(IntegrationPoints[g].Coordinates());
        double A = norm_2(n_gauss);
        n_gauss /= A;

        // Set nodal data
        array_1d<double, TNumNodes> nu_nodes;
        BoundedMatrix<double, TNumNodes, TDim> u_nodes;
        for (IndexType i = 0; i < TNumNodes; ++i) {
            nu_nodes[i] = rGeom[i].FastGetSolutionStepValue(DYNAMIC_VISCOSITY);
            const auto& r_v = rGeom[i].FastGetSolutionStepValue(VELOCITY);
            for (IndexType d = 0; d < TDim; ++d) {
                u_nodes(i, d) = r_v[d];
            }
        }

        const double delta_gauss = rCurrentProcessInfo.GetValue(TAUONE);
        
                const double crRightHandSideVector0 = DN(0,0)*u_nodes(0,1) - DN(0,1)*u_nodes(0,0) + DN(1,0)*u_nodes(1,1) - DN(1,1)*u_nodes(1,0) + DN(2,0)*u_nodes(2,1) - DN(2,1)*u_nodes(2,0);
        const double crRightHandSideVector1 = DN(0,0)*u_nodes(0,2) - DN(0,2)*u_nodes(0,0) + DN(1,0)*u_nodes(1,2) - DN(1,2)*u_nodes(1,0) + DN(2,0)*u_nodes(2,2) - DN(2,2)*u_nodes(2,0);
        const double crRightHandSideVector2 = DN(0,1)*u_nodes(0,2) - DN(0,2)*u_nodes(0,1) + DN(1,1)*u_nodes(1,2) - DN(1,2)*u_nodes(1,1) + DN(2,1)*u_nodes(2,2) - DN(2,2)*u_nodes(2,1);
        const double crRightHandSideVector3 = delta_gauss*w_g*(N[0]*nu_nodes[0] + N[1]*nu_nodes[1] + N[2]*nu_nodes[2]);
        rRightHandSideVector[0]+=0;
        rRightHandSideVector[1]+=0;
        rRightHandSideVector[2]+=0;
        rRightHandSideVector[3]+=crRightHandSideVector3*(crRightHandSideVector0*(DN(0,0)*n_gauss[1] - DN(0,1)*n_gauss[0]) + crRightHandSideVector1*(DN(0,0)*n_gauss[2] - DN(0,2)*n_gauss[0]) + crRightHandSideVector2*(DN(0,1)*n_gauss[2] - DN(0,2)*n_gauss[1]));
        rRightHandSideVector[4]+=0;
        rRightHandSideVector[5]+=0;
        rRightHandSideVector[6]+=0;
        rRightHandSideVector[7]+=crRightHandSideVector3*(crRightHandSideVector0*(DN(1,0)*n_gauss[1] - DN(1,1)*n_gauss[0]) + crRightHandSideVector1*(DN(1,0)*n_gauss[2] - DN(1,2)*n_gauss[0]) + crRightHandSideVector2*(DN(1,1)*n_gauss[2] - DN(1,2)*n_gauss[1]));
        rRightHandSideVector[8]+=0;
        rRightHandSideVector[9]+=0;
        rRightHandSideVector[10]+=0;
        rRightHandSideVector[11]+=crRightHandSideVector3*(crRightHandSideVector0*(DN(2,0)*n_gauss[1] - DN(2,1)*n_gauss[0]) + crRightHandSideVector1*(DN(2,0)*n_gauss[2] - DN(2,2)*n_gauss[0]) + crRightHandSideVector2*(DN(2,1)*n_gauss[2] - DN(2,2)*n_gauss[1]));
        

                const double crLeftHandSideMatrix0 = DN(0,0)*n_gauss[1] - DN(0,1)*n_gauss[0];
        const double crLeftHandSideMatrix1 = DN(0,0)*n_gauss[2] - DN(0,2)*n_gauss[0];
        const double crLeftHandSideMatrix2 = delta_gauss*w_g*(N[0]*nu_nodes[0] + N[1]*nu_nodes[1] + N[2]*nu_nodes[2]);
        const double crLeftHandSideMatrix3 = DN(0,1)*n_gauss[2] - DN(0,2)*n_gauss[1];
        const double crLeftHandSideMatrix4 = DN(1,0)*n_gauss[1] - DN(1,1)*n_gauss[0];
        const double crLeftHandSideMatrix5 = DN(1,0)*n_gauss[2] - DN(1,2)*n_gauss[0];
        const double crLeftHandSideMatrix6 = DN(1,1)*n_gauss[2] - DN(1,2)*n_gauss[1];
        const double crLeftHandSideMatrix7 = DN(2,0)*n_gauss[1] - DN(2,1)*n_gauss[0];
        const double crLeftHandSideMatrix8 = DN(2,0)*n_gauss[2] - DN(2,2)*n_gauss[0];
        const double crLeftHandSideMatrix9 = DN(2,1)*n_gauss[2] - DN(2,2)*n_gauss[1];
        rLeftHandSideMatrix(0,0)+=0;
        rLeftHandSideMatrix(0,1)+=0;
        rLeftHandSideMatrix(0,2)+=0;
        rLeftHandSideMatrix(0,3)+=0;
        rLeftHandSideMatrix(0,4)+=0;
        rLeftHandSideMatrix(0,5)+=0;
        rLeftHandSideMatrix(0,6)+=0;
        rLeftHandSideMatrix(0,7)+=0;
        rLeftHandSideMatrix(0,8)+=0;
        rLeftHandSideMatrix(0,9)+=0;
        rLeftHandSideMatrix(0,10)+=0;
        rLeftHandSideMatrix(0,11)+=0;
        rLeftHandSideMatrix(1,0)+=0;
        rLeftHandSideMatrix(1,1)+=0;
        rLeftHandSideMatrix(1,2)+=0;
        rLeftHandSideMatrix(1,3)+=0;
        rLeftHandSideMatrix(1,4)+=0;
        rLeftHandSideMatrix(1,5)+=0;
        rLeftHandSideMatrix(1,6)+=0;
        rLeftHandSideMatrix(1,7)+=0;
        rLeftHandSideMatrix(1,8)+=0;
        rLeftHandSideMatrix(1,9)+=0;
        rLeftHandSideMatrix(1,10)+=0;
        rLeftHandSideMatrix(1,11)+=0;
        rLeftHandSideMatrix(2,0)+=0;
        rLeftHandSideMatrix(2,1)+=0;
        rLeftHandSideMatrix(2,2)+=0;
        rLeftHandSideMatrix(2,3)+=0;
        rLeftHandSideMatrix(2,4)+=0;
        rLeftHandSideMatrix(2,5)+=0;
        rLeftHandSideMatrix(2,6)+=0;
        rLeftHandSideMatrix(2,7)+=0;
        rLeftHandSideMatrix(2,8)+=0;
        rLeftHandSideMatrix(2,9)+=0;
        rLeftHandSideMatrix(2,10)+=0;
        rLeftHandSideMatrix(2,11)+=0;
        rLeftHandSideMatrix(3,0)+=crLeftHandSideMatrix2*(DN(0,1)*crLeftHandSideMatrix0 + DN(0,2)*crLeftHandSideMatrix1);
        rLeftHandSideMatrix(3,1)+=-crLeftHandSideMatrix2*(DN(0,0)*crLeftHandSideMatrix0 - DN(0,2)*crLeftHandSideMatrix3);
        rLeftHandSideMatrix(3,2)+=-crLeftHandSideMatrix2*(DN(0,0)*crLeftHandSideMatrix1 + DN(0,1)*crLeftHandSideMatrix3);
        rLeftHandSideMatrix(3,3)+=0;
        rLeftHandSideMatrix(3,4)+=crLeftHandSideMatrix2*(DN(1,1)*crLeftHandSideMatrix0 + DN(1,2)*crLeftHandSideMatrix1);
        rLeftHandSideMatrix(3,5)+=-crLeftHandSideMatrix2*(DN(1,0)*crLeftHandSideMatrix0 - DN(1,2)*crLeftHandSideMatrix3);
        rLeftHandSideMatrix(3,6)+=-crLeftHandSideMatrix2*(DN(1,0)*crLeftHandSideMatrix1 + DN(1,1)*crLeftHandSideMatrix3);
        rLeftHandSideMatrix(3,7)+=0;
        rLeftHandSideMatrix(3,8)+=crLeftHandSideMatrix2*(DN(2,1)*crLeftHandSideMatrix0 + DN(2,2)*crLeftHandSideMatrix1);
        rLeftHandSideMatrix(3,9)+=-crLeftHandSideMatrix2*(DN(2,0)*crLeftHandSideMatrix0 - DN(2,2)*crLeftHandSideMatrix3);
        rLeftHandSideMatrix(3,10)+=-crLeftHandSideMatrix2*(DN(2,0)*crLeftHandSideMatrix1 + DN(2,1)*crLeftHandSideMatrix3);
        rLeftHandSideMatrix(3,11)+=0;
        rLeftHandSideMatrix(4,0)+=0;
        rLeftHandSideMatrix(4,1)+=0;
        rLeftHandSideMatrix(4,2)+=0;
        rLeftHandSideMatrix(4,3)+=0;
        rLeftHandSideMatrix(4,4)+=0;
        rLeftHandSideMatrix(4,5)+=0;
        rLeftHandSideMatrix(4,6)+=0;
        rLeftHandSideMatrix(4,7)+=0;
        rLeftHandSideMatrix(4,8)+=0;
        rLeftHandSideMatrix(4,9)+=0;
        rLeftHandSideMatrix(4,10)+=0;
        rLeftHandSideMatrix(4,11)+=0;
        rLeftHandSideMatrix(5,0)+=0;
        rLeftHandSideMatrix(5,1)+=0;
        rLeftHandSideMatrix(5,2)+=0;
        rLeftHandSideMatrix(5,3)+=0;
        rLeftHandSideMatrix(5,4)+=0;
        rLeftHandSideMatrix(5,5)+=0;
        rLeftHandSideMatrix(5,6)+=0;
        rLeftHandSideMatrix(5,7)+=0;
        rLeftHandSideMatrix(5,8)+=0;
        rLeftHandSideMatrix(5,9)+=0;
        rLeftHandSideMatrix(5,10)+=0;
        rLeftHandSideMatrix(5,11)+=0;
        rLeftHandSideMatrix(6,0)+=0;
        rLeftHandSideMatrix(6,1)+=0;
        rLeftHandSideMatrix(6,2)+=0;
        rLeftHandSideMatrix(6,3)+=0;
        rLeftHandSideMatrix(6,4)+=0;
        rLeftHandSideMatrix(6,5)+=0;
        rLeftHandSideMatrix(6,6)+=0;
        rLeftHandSideMatrix(6,7)+=0;
        rLeftHandSideMatrix(6,8)+=0;
        rLeftHandSideMatrix(6,9)+=0;
        rLeftHandSideMatrix(6,10)+=0;
        rLeftHandSideMatrix(6,11)+=0;
        rLeftHandSideMatrix(7,0)+=crLeftHandSideMatrix2*(DN(0,1)*crLeftHandSideMatrix4 + DN(0,2)*crLeftHandSideMatrix5);
        rLeftHandSideMatrix(7,1)+=-crLeftHandSideMatrix2*(DN(0,0)*crLeftHandSideMatrix4 - DN(0,2)*crLeftHandSideMatrix6);
        rLeftHandSideMatrix(7,2)+=-crLeftHandSideMatrix2*(DN(0,0)*crLeftHandSideMatrix5 + DN(0,1)*crLeftHandSideMatrix6);
        rLeftHandSideMatrix(7,3)+=0;
        rLeftHandSideMatrix(7,4)+=crLeftHandSideMatrix2*(DN(1,1)*crLeftHandSideMatrix4 + DN(1,2)*crLeftHandSideMatrix5);
        rLeftHandSideMatrix(7,5)+=-crLeftHandSideMatrix2*(DN(1,0)*crLeftHandSideMatrix4 - DN(1,2)*crLeftHandSideMatrix6);
        rLeftHandSideMatrix(7,6)+=-crLeftHandSideMatrix2*(DN(1,0)*crLeftHandSideMatrix5 + DN(1,1)*crLeftHandSideMatrix6);
        rLeftHandSideMatrix(7,7)+=0;
        rLeftHandSideMatrix(7,8)+=crLeftHandSideMatrix2*(DN(2,1)*crLeftHandSideMatrix4 + DN(2,2)*crLeftHandSideMatrix5);
        rLeftHandSideMatrix(7,9)+=-crLeftHandSideMatrix2*(DN(2,0)*crLeftHandSideMatrix4 - DN(2,2)*crLeftHandSideMatrix6);
        rLeftHandSideMatrix(7,10)+=-crLeftHandSideMatrix2*(DN(2,0)*crLeftHandSideMatrix5 + DN(2,1)*crLeftHandSideMatrix6);
        rLeftHandSideMatrix(7,11)+=0;
        rLeftHandSideMatrix(8,0)+=0;
        rLeftHandSideMatrix(8,1)+=0;
        rLeftHandSideMatrix(8,2)+=0;
        rLeftHandSideMatrix(8,3)+=0;
        rLeftHandSideMatrix(8,4)+=0;
        rLeftHandSideMatrix(8,5)+=0;
        rLeftHandSideMatrix(8,6)+=0;
        rLeftHandSideMatrix(8,7)+=0;
        rLeftHandSideMatrix(8,8)+=0;
        rLeftHandSideMatrix(8,9)+=0;
        rLeftHandSideMatrix(8,10)+=0;
        rLeftHandSideMatrix(8,11)+=0;
        rLeftHandSideMatrix(9,0)+=0;
        rLeftHandSideMatrix(9,1)+=0;
        rLeftHandSideMatrix(9,2)+=0;
        rLeftHandSideMatrix(9,3)+=0;
        rLeftHandSideMatrix(9,4)+=0;
        rLeftHandSideMatrix(9,5)+=0;
        rLeftHandSideMatrix(9,6)+=0;
        rLeftHandSideMatrix(9,7)+=0;
        rLeftHandSideMatrix(9,8)+=0;
        rLeftHandSideMatrix(9,9)+=0;
        rLeftHandSideMatrix(9,10)+=0;
        rLeftHandSideMatrix(9,11)+=0;
        rLeftHandSideMatrix(10,0)+=0;
        rLeftHandSideMatrix(10,1)+=0;
        rLeftHandSideMatrix(10,2)+=0;
        rLeftHandSideMatrix(10,3)+=0;
        rLeftHandSideMatrix(10,4)+=0;
        rLeftHandSideMatrix(10,5)+=0;
        rLeftHandSideMatrix(10,6)+=0;
        rLeftHandSideMatrix(10,7)+=0;
        rLeftHandSideMatrix(10,8)+=0;
        rLeftHandSideMatrix(10,9)+=0;
        rLeftHandSideMatrix(10,10)+=0;
        rLeftHandSideMatrix(10,11)+=0;
        rLeftHandSideMatrix(11,0)+=crLeftHandSideMatrix2*(DN(0,1)*crLeftHandSideMatrix7 + DN(0,2)*crLeftHandSideMatrix8);
        rLeftHandSideMatrix(11,1)+=-crLeftHandSideMatrix2*(DN(0,0)*crLeftHandSideMatrix7 - DN(0,2)*crLeftHandSideMatrix9);
        rLeftHandSideMatrix(11,2)+=-crLeftHandSideMatrix2*(DN(0,0)*crLeftHandSideMatrix8 + DN(0,1)*crLeftHandSideMatrix9);
        rLeftHandSideMatrix(11,3)+=0;
        rLeftHandSideMatrix(11,4)+=crLeftHandSideMatrix2*(DN(1,1)*crLeftHandSideMatrix7 + DN(1,2)*crLeftHandSideMatrix8);
        rLeftHandSideMatrix(11,5)+=-crLeftHandSideMatrix2*(DN(1,0)*crLeftHandSideMatrix7 - DN(1,2)*crLeftHandSideMatrix9);
        rLeftHandSideMatrix(11,6)+=-crLeftHandSideMatrix2*(DN(1,0)*crLeftHandSideMatrix8 + DN(1,1)*crLeftHandSideMatrix9);
        rLeftHandSideMatrix(11,7)+=0;
        rLeftHandSideMatrix(11,8)+=crLeftHandSideMatrix2*(DN(2,1)*crLeftHandSideMatrix7 + DN(2,2)*crLeftHandSideMatrix8);
        rLeftHandSideMatrix(11,9)+=-crLeftHandSideMatrix2*(DN(2,0)*crLeftHandSideMatrix7 - DN(2,2)*crLeftHandSideMatrix9);
        rLeftHandSideMatrix(11,10)+=-crLeftHandSideMatrix2*(DN(2,0)*crLeftHandSideMatrix8 + DN(2,1)*crLeftHandSideMatrix9);
        rLeftHandSideMatrix(11,11)+=0;
        
    }
}

template class CustomCurlCondition<2,2>;
template class CustomCurlCondition<3,3>;

} // namespace Kratos