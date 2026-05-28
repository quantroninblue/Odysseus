/*
association.cpp
*/

#include "association.hpp"

#include <cmath>
#include <limits>
#include <set>


float Association::computeDistance(

    const TrackState& track,

    const DetectionMeasurement& detection
)
{
    float dx =
        track.center_x -
        detection.center_x;

    float dy =
        track.center_y -
        detection.center_y;

    return std::sqrt(
        dx * dx +
        dy * dy
    );
}


AssociationResult
Association::associateTracksAndDetections(

    const std::vector<TrackState>& tracks,

    const std::vector<DetectionMeasurement>& detections,

    float max_distance
)
{
    AssociationResult result;

    // ------------------------------------------------------------
    // Empty cases
    // ------------------------------------------------------------

    if (tracks.empty())
    {
        for (int i = 0; i < (int)detections.size(); i++)
        {
            result.unmatched_detections.push_back(i);
        }

        return result;
    }

    if (detections.empty())
    {
        for (int i = 0; i < (int)tracks.size(); i++)
        {
            result.unmatched_tracks.push_back(i);
        }

        return result;
    }

    // ------------------------------------------------------------
    // Greedy nearest-neighbor matching
    // ------------------------------------------------------------

    std::set<int> used_tracks;

    std::set<int> used_detections;

    for (int det_idx = 0;
         det_idx < (int)detections.size();
         det_idx++)
    {
        int best_track_idx = -1;

        float best_distance =
            std::numeric_limits<float>::max();

        for (int track_idx = 0;
             track_idx < (int)tracks.size();
             track_idx++)
        {
            if (used_tracks.count(track_idx))
            {
                continue;
            }

            float distance =
                computeDistance(
                    tracks[track_idx],
                    detections[det_idx]
                );

            if (
                distance < best_distance &&
                distance < max_distance
            )
            {
                best_distance = distance;

                best_track_idx = track_idx;
            }
        }

        if (best_track_idx >= 0)
        {
            AssociationMatch match;

            match.track_idx =
                best_track_idx;

            match.detection_idx =
                det_idx;

            result.matches.push_back(match);

            used_tracks.insert(
                best_track_idx
            );

            used_detections.insert(
                det_idx
            );
        }
    }

    // ------------------------------------------------------------
    // Unmatched tracks
    // ------------------------------------------------------------

    for (int i = 0;
         i < (int)tracks.size();
         i++)
    {
        if (!used_tracks.count(i))
        {
            result.unmatched_tracks.push_back(i);
        }
    }

    // ------------------------------------------------------------
    // Unmatched detections
    // ------------------------------------------------------------

    for (int i = 0;
         i < (int)detections.size();
         i++)
    {
        if (!used_detections.count(i))
        {
            result.unmatched_detections.push_back(i);
        }
    }

    return result;
}