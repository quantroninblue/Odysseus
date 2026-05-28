/*
tracking_validation.cpp
*/

#include <iostream>

#include "tracking_validation.hpp"
#include "track_state.hpp"


void TrackingValidation::run()
{
    std::cout << "\n=== TrackState Validation ===\n";

    // ------------------------------------------------------------
    // Create track
    // ------------------------------------------------------------

    TrackState track(

        1,

        320.0f,
        240.0f,

        0.0f,

        120.0f,
        80.0f
    );

    std::cout << "\nInitial Track:\n";

    std::cout
        << "Track ID: " << track.track_id << "\n"
        << "Center: ("
        << track.center_x << ", "
        << track.center_y << ")\n"
        << "Yaw: " << track.yaw_rad << "\n"
        << "Dimensions: "
        << track.width_px << " x "
        << track.height_px << "\n";

    // ------------------------------------------------------------
    // Simulated updates
    // ------------------------------------------------------------

    float positions[4][2] = {

        {325.0f, 242.0f},
        {330.0f, 245.0f},
        {338.0f, 249.0f},
        {345.0f, 255.0f}
    };

    for (int i = 0; i < 4; i++)
    {
        track.update(

            positions[i][0],
            positions[i][1],

            0.05f * i,

            120.0f + i,
            80.0f + i
        );

        std::cout
            << "\nUpdate " << i + 1 << ":\n";

        std::cout
            << "Center: ("
            << track.center_x << ", "
            << track.center_y << ")\n";

        std::cout
            << "Velocity: ("
            << track.velocity_x << ", "
            << track.velocity_y << ")\n";

        std::cout
            << "Age: "
            << track.age << "\n";

        std::cout
            << "History Size: "
            << track.history.size() << "\n";
    }

    // ------------------------------------------------------------
    // Missed frames
    // ------------------------------------------------------------

    std::cout
        << "\nSimulating missed frames...\n";

    track.markMissed();

    track.markMissed();

    std::cout
        << "Missed Frames: "
        << track.missed_frames << "\n";

    // ------------------------------------------------------------
    // Deactivation
    // ------------------------------------------------------------

    std::cout
        << "\nDeactivating track...\n";

    track.deactivate();

    std::cout
        << "Active: "
        << track.is_active << "\n";

    std::cout
        << "\nValidation complete.\n";
}