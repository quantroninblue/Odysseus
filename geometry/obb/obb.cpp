// obb.cpp

#include "obb.hpp"

#include <cmath>
#include <algorithm>


OBBEstimator::OBBEstimator(
    double fx,
    double fy
)
{
    fx_ = fx;
    fy_ = fy;
}


OBBResult OBBEstimator::estimateOBB(
    const std::vector<Point2D>& contour,
    double depth_m
)
{
    OBBResult result;

    if (contour.empty())
    {
        return result;
    }

    int min_x = contour[0].x;
    int max_x = contour[0].x;

    int min_y = contour[0].y;
    int max_y = contour[0].y;

    for (const auto& p : contour)
    {
        min_x = std::min(min_x, p.x);
        max_x = std::max(max_x, p.x);

        min_y = std::min(min_y, p.y);
        max_y = std::max(max_y, p.y);
    }

    double width_px  = max_x - min_x;
    double height_px = max_y - min_y;

    result.center_x = (min_x + max_x) / 2.0;
    result.center_y = (min_y + max_y) / 2.0;

    result.width_px  = width_px;
    result.height_px = height_px;

    result.width_m =
        width_px *
        depth_m /
        fx_;

    result.height_m =
        height_px *
        depth_m /
        fy_;

    result.angle_deg = 0.0;
    result.yaw_rad   = 0.0;

    result.contour_area =
        width_px * height_px;

    result.box_points = {
        {min_x, min_y},
        {max_x, min_y},
        {max_x, max_y},
        {min_x, max_y}
    };

    return result;
}