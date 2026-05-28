#pragma once

/*
kalman_filter.hpp

Constant-velocity Kalman filter for 2D tracking.

State:
[x, y, vx, vy]
*/

#include <array>


class KalmanFilter2D
{
private:

    // ------------------------------------------------------------
    // State vector
    // ------------------------------------------------------------

    std::array<double, 4> x;

    // ------------------------------------------------------------
    // Timing
    // ------------------------------------------------------------

    double dt;

    // ------------------------------------------------------------
    // Noise parameters
    // ------------------------------------------------------------

    double process_noise;

    double measurement_noise;

public:

    KalmanFilter2D(

        double dt = 1.0,

        double process_noise = 1e-2,

        double measurement_noise = 1e-1
    );

    // ------------------------------------------------------------
    // Initialization
    // ------------------------------------------------------------

    void initialize(
        double x,
        double y
    );

    // ------------------------------------------------------------
    // Prediction
    // ------------------------------------------------------------

    void predict();

    // ------------------------------------------------------------
    // Measurement update
    // ------------------------------------------------------------

    void update(
        double measured_x,
        double measured_y
    );

    // ------------------------------------------------------------
    // State accessors
    // ------------------------------------------------------------

    double getX() const;

    double getY() const;

    double getVX() const;

    double getVY() const;
};