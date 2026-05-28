// backprojection.cpp

#include "backprojection.hpp"

#include <iostream>
#include <cmath>

Backprojector::Backprojector(
    double fx,
    double fy,
    double cx,
    double cy,
    int cloud_stride
)
{
    fx_ = fx;
    fy_ = fy;
    cx_ = cx;
    cy_ = cy;

    cloud_stride_ = cloud_stride;
}

std::vector<Point3D> Backprojector::backprojectROIToWorld(
    const std::vector<std::vector<uint16_t>>& depth_frame,

    int img_x1,
    int img_y1,
    int img_x2,
    int img_y2,

    const std::vector<std::vector<double>>& tf_matrix,

    double depth_min_mm,
    double depth_max_mm,

    int img_w,
    int img_h
)
{
    std::vector<Point3D> cloud;

    int dep_h = depth_frame.size();
    int dep_w = depth_frame[0].size();

    double sx = static_cast<double>(dep_w) / img_w;
    double sy = static_cast<double>(dep_h) / img_h;

    int dx1 = std::max(
        0,
        static_cast<int>(img_x1 * sx)
    );

    int dy1 = std::max(
        0,
        static_cast<int>(img_y1 * sy)
    );

    int dx2 = std::min(
        dep_w,
        static_cast<int>(img_x2 * sx)
    );

    int dy2 = std::min(
        dep_h,
        static_cast<int>(img_y2 * sy)
    );

    for (
        int v = dy1;
        v < dy2;
        v += cloud_stride_
    )
    {
        for (
            int u = dx1;
            u < dx2;
            u += cloud_stride_
        )
        {
            double depth_mm = depth_frame[v][u];

            if (
                depth_mm < depth_min_mm ||
                depth_mm > depth_max_mm
            )
            {
                continue;
            }

            double d_m = depth_mm / 1000.0;

            double u_rgb = u / sx;
            double v_rgb = v / sy;

            double X_cam =
                (u_rgb - cx_) *
                d_m / fx_;

            double Y_cam =
                (v_rgb - cy_) *
                d_m / fy_;

            double Z_cam = d_m;

            double X_world =
                tf_matrix[0][0] * X_cam +
                tf_matrix[0][1] * Y_cam +
                tf_matrix[0][2] * Z_cam +
                tf_matrix[0][3];

            double Y_world =
                tf_matrix[1][0] * X_cam +
                tf_matrix[1][1] * Y_cam +
                tf_matrix[1][2] * Z_cam +
                tf_matrix[1][3];

            double Z_world =
                tf_matrix[2][0] * X_cam +
                tf_matrix[2][1] * Y_cam +
                tf_matrix[2][2] * Z_cam +
                tf_matrix[2][3];

            Point3D p;

            p.x = X_world;
            p.y = Y_world;
            p.z = Z_world;

            cloud.push_back(p);
        }
    }

    return cloud;
}