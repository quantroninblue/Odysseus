/*
track_state.cpp
*/

#include "track_state.hpp"


TrackState::TrackState(

    int track_id,

    float center_x,
    float center_y,

    float yaw_rad,

    float width_px,
    float height_px,

    float confidence,

    int history_size
)
{
    this->track_id = track_id;

    // ------------------------------------------------------------
    // Temporal bookkeeping
    // ------------------------------------------------------------

    this->age = 1;

    this->missed_frames = 0;

    this->last_seen_frame = 0;

    this->is_active = true;

    // ------------------------------------------------------------
    // Geometry
    // ------------------------------------------------------------

    this->center_x = center_x;
    this->center_y = center_y;

    this->yaw_rad = yaw_rad;

    this->width_px = width_px;
    this->height_px = height_px;

    // ------------------------------------------------------------
    // Velocity
    // ------------------------------------------------------------

    this->velocity_x = 0.0f;
    this->velocity_y = 0.0f;

    // ------------------------------------------------------------
    // Confidence
    // ------------------------------------------------------------

    this->confidence = confidence;

    // ------------------------------------------------------------
    // History
    // ------------------------------------------------------------

    this->history_size = history_size;

    appendHistory();
}


void TrackState::appendHistory()
{
    HistoryEntry entry;

    entry.center_x = center_x;
    entry.center_y = center_y;

    entry.yaw_rad = yaw_rad;

    entry.width_px = width_px;
    entry.height_px = height_px;

    history.push_back(entry);

    while ((int)history.size() > history_size)
    {
        history.pop_front();
    }
}


void TrackState::update(

    float center_x,
    float center_y,

    float yaw_rad,

    float width_px,
    float height_px,

    float confidence
)
{
    // ------------------------------------------------------------
    // Velocity update
    // ------------------------------------------------------------

    velocity_x = center_x - this->center_x;

    velocity_y = center_y - this->center_y;

    // ------------------------------------------------------------
    // State update
    // ------------------------------------------------------------

    this->center_x = center_x;
    this->center_y = center_y;

    this->yaw_rad = yaw_rad;

    this->width_px = width_px;
    this->height_px = height_px;

    this->confidence = confidence;

    // ------------------------------------------------------------
    // Temporal bookkeeping
    // ------------------------------------------------------------

    age += 1;

    missed_frames = 0;

    is_active = true;

    // ------------------------------------------------------------
    // History
    // ------------------------------------------------------------

    appendHistory();
}


void TrackState::markMissed()
{
    missed_frames += 1;
}


void TrackState::deactivate()
{
    is_active = false;
}