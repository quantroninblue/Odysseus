#pragma once

/*
tracker_reference.hpp

Persistent multi-object tracking runtime.
*/

#include <vector>

#include "track_state.hpp"
#include "association.hpp"


class MultiObjectTracker
{
private:

    // ------------------------------------------------------------
    // Active tracks
    // ------------------------------------------------------------

    std::vector<TrackState> tracks;

    // ------------------------------------------------------------
    // ID generation
    // ------------------------------------------------------------

    int next_track_id;

    // ------------------------------------------------------------
    // Lifecycle configuration
    // ------------------------------------------------------------

    int max_missed_frames;

    float association_distance;

    // ------------------------------------------------------------
    // Runtime bookkeeping
    // ------------------------------------------------------------

    int frame_index;

public:

    MultiObjectTracker(

        int max_missed_frames = 10,

        float association_distance = 100.0f
    );

    void createTrack(
        const DetectionMeasurement& detection
    );

    void updateTrack(

        TrackState& track,

        const DetectionMeasurement& detection
    );

    void handleUnmatchedTracks(
        const std::vector<int>& unmatched_track_indices
    );

    void removeDeadTracks();

    std::vector<TrackState>& update(
        const std::vector<DetectionMeasurement>& detections
    );

    const std::vector<TrackState>& getTracks() const;
};