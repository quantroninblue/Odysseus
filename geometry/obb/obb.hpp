// obb.hpp

#ifndef OBB_HPP
#define OBB_HPP

#include <vector>


struct Point2D
{
    int x;
    int y;
};


struct OBBResult
{
    double center_x;
    double center_y;

    double width_px;
    double height_px;

    double width_m;
    double height_m;

    double angle_deg;
    double yaw_rad;

    double contour_area;

    std::vector<Point2D> box_points;
};


class OBBEstimator
{
public:

    OBBEstimator(
        double fx,
        double fy
    );

    OBBResult estimateOBB(
        const std::vector<Point2D>& contour,
        double depth_m
    );

private:

    double fx_;
    double fy_;
};

#endif