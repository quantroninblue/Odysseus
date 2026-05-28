#include "transforms.hpp"

#include <Eigen/Dense>


namespace transforms
{
    Eigen::Matrix4d tfToMatrix(
        double tx,
        double ty,
        double tz,
        double qx,
        double qy,
        double qz,
        double qw
    )
    {
        Eigen::Matrix3d rotation_matrix;

        rotation_matrix <<
            1 - 2 * (qy * qy + qz * qz),
            2 * (qx * qy - qz * qw),
            2 * (qx * qz + qy * qw),

            2 * (qx * qy + qz * qw),
            1 - 2 * (qx * qx + qz * qz),
            2 * (qy * qz - qx * qw),

            2 * (qx * qz - qy * qw),
            2 * (qy * qz + qx * qw),
            1 - 2 * (qx * qx + qy * qy);

        Eigen::Matrix4d transform_matrix =
            Eigen::Matrix4d::Identity();

        transform_matrix.block<3,3>(0,0) =
            rotation_matrix;

        transform_matrix(0,3) = tx;
        transform_matrix(1,3) = ty;
        transform_matrix(2,3) = tz;

        return transform_matrix;
    }
}