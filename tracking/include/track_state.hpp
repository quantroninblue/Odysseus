#pragma once

/*
track_state.hpp

Persistent tracked object state representation.
*/

#include <deque>


struct HistoryEntry
{
    float center_x;
    float center_y;

    float yaw_rad;

    float width_px;
    float height_px;
};


class TrackState
{
public:

    // ------------------------------------------------------------
    // Identity
    // ------------------------------------------------------------

    int track_id;

    // ------------------------------------------------------------
    // Temporal bookkeeping
    // ------------------------------------------------------------

    int age;

    int missed_frames;

    int last_seen_frame;

    bool is_active;

    // ------------------------------------------------------------
    // Geometry
    // ------------------------------------------------------------

    float center_x;
    float center_y;

    float yaw_rad;

    float width_px;
    float height_px;

    // ------------------------------------------------------------
    // Velocity
    // ------------------------------------------------------------

    float velocity_x;
    float velocity_y;

    // ------------------------------------------------------------
    // Confidence
    // ------------------------------------------------------------

    float confidence;

    // ------------------------------------------------------------
    // Temporal history
    // ------------------------------------------------------------

    std::deque<HistoryEntry> history;

    int history_size;

public:

    TrackState(

        int track_id,

        float center_x,
        float center_y,

        float yaw_rad,

        float width_px,
        float height_px,

        float confidence = 1.0f,

        int history_size = 30
    );

    void update(

        float center_x,
        float center_y,

        float yaw_rad,

        float width_px,
        float height_px,

        float confidence = 1.0f
    );

    void markMissed();

    void deactivate();

    void appendHistory();
};