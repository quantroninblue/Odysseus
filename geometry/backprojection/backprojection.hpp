// backprojection.hpp

#ifndef BACKPROJECTION_HPP
#define BACKPROJECTION_HPP

#include <vector>

struct Point3D
{
    double x;
    double y;
    double z;
};

class Backprojector
{
public:

    Backprojector(
        double fx,
        double fy,
        double cx,
        double cy,
        int cloud_stride = 4
    );

    std::vector<Point3D> backprojectROIToWorld(
        const std::vector<std::vector<uint16_t>>& depth_frame,

        int img_x1,
        int img_y1,
        int img_x2,
        int img_y2,

        const std::vector<std::vector<double>>& tf_matrix,

        double depth_min_mm,
        double depth_max_mm,

        int img_w = 640,
        int img_h = 640
    );

private:

    double fx_;
    double fy_;
    double cx_;
    double cy_;

    int cloud_stride_;
};

#endif