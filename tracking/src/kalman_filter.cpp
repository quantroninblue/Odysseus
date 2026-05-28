/*
kalman_filter.cpp

Constant-velocity Kalman filter for 2D tracking.
*/

#include "kalman_filter.hpp"

#include <iostream>


KalmanFilter2D::KalmanFilter2D(

    double dt,

    double process_noise,

    double measurement_noise
)
{
    this->dt = dt;

    this->process_noise =
        process_noise;

    this->measurement_noise =
        measurement_noise;

    this->x = {
        0.0,
        0.0,
        0.0,
        0.0
    };
}


void KalmanFilter2D::initialize(
    double x,
    double y
)
{
    this->x[0] = x;
    this->x[1] = y;

    this->x[2] = 0.0;
    this->x[3] = 0.0;
}


void KalmanFilter2D::predict()
{
    // ------------------------------------------------------------
    // Constant velocity prediction
    // ------------------------------------------------------------

    this->x[0] += this->x[2] * dt;

    this->x[1] += this->x[3] * dt;
}


void KalmanFilter2D::update(
    double measured_x,
    double measured_y
)
{
    // ------------------------------------------------------------
    // Residual
    // ------------------------------------------------------------

    double residual_x =
        measured_x - this->x[0];

    double residual_y =
        measured_y - this->x[1];

    // ------------------------------------------------------------
    // Simple smoothing gain
    // ------------------------------------------------------------

    double gain = 0.5;

    // ------------------------------------------------------------
    // Velocity update
    // ------------------------------------------------------------

    this->x[2] =
        residual_x / dt;

    this->x[3] =
        residual_y / dt;

    // ------------------------------------------------------------
    // Position update
    // ------------------------------------------------------------

    this->x[0] +=
        gain * residual_x;

    this->x[1] +=
        gain * residual_y;
}


double KalmanFilter2D::getX() const
{
    return this->x[0];
}


double KalmanFilter2D::getY() const
{
    return this->x[1];
}


double KalmanFilter2D::getVX() const
{
    return this->x[2];
}


double KalmanFilter2D::getVY() const
{
    return this->x[3];
}