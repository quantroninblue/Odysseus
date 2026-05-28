/*
tracker_reference.cpp
*/

#include "tracker_reference.hpp"


MultiObjectTracker::MultiObjectTracker(

    int max_missed_frames,

    float association_distance
)
{
    this->next_track_id = 0;

    this->max_missed_frames =
        max_missed_frames;

    this->association_distance =
        association_distance;

    this->frame_index = 0;
}


void MultiObjectTracker::createTrack(
    const DetectionMeasurement& detection
)
{
    TrackState track(

        next_track_id,

        detection.center_x,
        detection.center_y,

        detection.yaw_rad,

        detection.width_px,
        detection.height_px
    );

    next_track_id++;

    tracks.push_back(track);
}


void MultiObjectTracker::updateTrack(

    TrackState& track,

    const DetectionMeasurement& detection
)
{
    track.update(

        detection.center_x,
        detection.center_y,

        detection.yaw_rad,

        detection.width_px,
        detection.height_px
    );

    track.last_seen_frame =
        frame_index;
}


void MultiObjectTracker::handleUnmatchedTracks(
    const std::vector<int>& unmatched_track_indices
)
{
    for (int idx : unmatched_track_indices)
    {
        tracks[idx].markMissed();
    }
}


void MultiObjectTracker::removeDeadTracks()
{
    std::vector<TrackState> surviving_tracks;

    for (auto& track : tracks)
    {
        if (
            track.missed_frames <=
            max_missed_frames
        )
        {
            surviving_tracks.push_back(track);
        }
        else
        {
            track.deactivate();
        }
    }

    tracks = surviving_tracks;
}


std::vector<TrackState>&
MultiObjectTracker::update(
    const std::vector<DetectionMeasurement>& detections
)
{
    frame_index++;

    // ------------------------------------------------------------
    // No tracks yet
    // ------------------------------------------------------------

    if (tracks.empty())
    {
        for (const auto& detection : detections)
        {
            createTrack(detection);
        }

        return tracks;
    }

    // ------------------------------------------------------------
    // Association
    // ------------------------------------------------------------

    AssociationResult association_result =
        Association::associateTracksAndDetections(

            tracks,

            detections,

            association_distance
        );

    // ------------------------------------------------------------
    // Matched updates
    // ------------------------------------------------------------

    for (const auto& match :
         association_result.matches)
    {
        updateTrack(

            tracks[match.track_idx],

            detections[match.detection_idx]
        );
    }

    // ------------------------------------------------------------
    // Unmatched tracks
    // ------------------------------------------------------------

    handleUnmatchedTracks(
        association_result.unmatched_tracks
    );

    // ------------------------------------------------------------
    // New tracks
    // ------------------------------------------------------------

    for (int det_idx :
         association_result.unmatched_detections)
    {
        createTrack(
            detections[det_idx]
        );
    }

    // ------------------------------------------------------------
    // Cleanup
    // ------------------------------------------------------------

    removeDeadTracks();

    return tracks;
}


const std::vector<TrackState>&
MultiObjectTracker::getTracks() const
{
    return tracks;
}