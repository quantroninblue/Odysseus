#pragma once

#include <Eigen/Dense>


namespace transforms
{
    /**
     * Convert translation + quaternion into a 4x4 homogeneous transform matrix.
     *
     * Parameters
     * ----------
     * tx, ty, tz :
     *     Translation components.
     *
     * qx, qy, qz, qw :
     *     Quaternion components.
     *
     * Returns
     * -------
     * Eigen::Matrix4d
     *     4x4 homogeneous transform matrix.
     */
    Eigen::Matrix4d tfToMatrix(
        double tx,
        double ty,
        double tz,
        double qx,
        double qy,
        double qz,
        double qw
    );
}