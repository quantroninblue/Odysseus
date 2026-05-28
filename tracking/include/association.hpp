#pragma once

/*
association.hpp

Detection-to-track association logic.
*/

#include <vector>

#include "track_state.hpp"


struct DetectionMeasurement
{
    float center_x;
    float center_y;

    float yaw_rad;

    float width_px;
    float height_px;
};


struct AssociationMatch
{
    int track_idx;
    int detection_idx;
};


struct AssociationResult
{
    std::vector<AssociationMatch> matches;

    std::vector<int> unmatched_tracks;

    std::vector<int> unmatched_detections;
};


class Association
{
public:

    static float computeDistance(

        const TrackState& track,

        const DetectionMeasurement& detection
    );

    static AssociationResult associateTracksAndDetections(

        const std::vector<TrackState>& tracks,

        const std::vector<DetectionMeasurement>& detections,

        float max_distance = 100.0f
    );
};