#include <iostream>
#include "Eigen/Eigen"

using namespace std;
using namespace Eigen;

bool SolveSystems(const Matrix2d& A,
                  double& detA,
                  double& condA,
                  const Vector2d& b,
                  double& errRelPA,
                  double& errRelQR)
{
    JacobiSVD<Matrix2d> svd(A);
    Vector2d singularValuesA=svd.singularValues();
    condA=singularValuesA.maxCoeff()/singularValuesA.minCoeff();
    detA=A.determinant();
    if (singularValuesA.minCoeff()<1e-16)
    {
        errRelPA=-1;
        errRelQR=-1;
        return false;
    }
    Vector2d exactSolution(2);
    exactSolution << -1.0e+0,
        -1.0e+00;

    Vector2d xPA= A.fullPivLu().solve(b);
    Vector2d xQR= A.colPivHouseholderQr().solve(b);

    errRelPA= (exactSolution-xPA).norm()/exactSolution.norm();
    errRelQR= (exactSolution-xQR).norm() / exactSolution.norm();
    return true;
}



int main()
{
    Matrix2d A1;
    A1<<5.547001962252291e-01, -3.770900990025203e-02,
        8.320502943378437e-01,-9.992887623566787e-01;
    Vector2d b1(2);
    b1<<-5.169911863249772e-01,
        1.672384680188350e-01;
    double detA1,condA1,errRelPA1,errRelQRA1;
    if(SolveSystems(A1,detA1,condA1,b1,errRelPA1,errRelQRA1))
        cout<<scientific<<"Case 1:  "<<"Relative error PA="<<errRelPA1<<"        "<<"Relative error QR="<<errRelQRA1<<endl;
    else
        cout<< "Matrix is singular";



    Matrix2d A2;
    A2<<5.547001962252291e-01, -5.540607316466765e-01,
        8.320502943378437e-01,-8.324762492991313e-01;
    Vector2d b2(2);
    b2<<-6.394645785530173e-04,
        4.259549612877223e-04;
    double detA2,condA2,errRelPA2,errRelQRA2;
    if(SolveSystems(A2,detA2,condA2,b2,errRelPA2,errRelQRA2))
        cout<<scientific<<"Case 2 :  "<<"Relative error PA="<<errRelPA2<<"        "<<"Relative error QR="<<errRelQRA2<<endl;
    else
        cout<< "Matrix is singular";

    Matrix2d A3;
    A3<<5.547001962252291e-01, -5.547001955851905e-01,
        8.320502943378437e-01,    -8.320502947645361e-01;
    Vector2d b3(2);
    b3<<-6.400391328043042e-10,
        4.266924591433963e-10;
    double detA3,condA3,errRelPA3,errRelQRA3;
    if(SolveSystems(A3,detA3,condA3,b3,errRelPA3,errRelQRA3))
        cout<<scientific<<"Case 3:  "<<"Relative error PA="<<errRelPA3<<"        "<<"Relative error QR="<<errRelQRA3<<endl;
    else
        cout<< "Matrix is singular";



    return 0;
}
