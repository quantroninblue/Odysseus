/*
replay_validation.cpp

Replay-driven modular integration validation runtime.
*/

#include "replay_validation.hpp"

#include <iostream>


ReplayValidationRuntime::ReplayValidationRuntime(

    const std::string& rgb_video_path,

    const std::string& depth_video_path
)
{
    this->rgb_video_path =
        rgb_video_path;

    this->depth_video_path =
        depth_video_path;

    this->export_directory =
        "replay_validation/exports";

    this->export_video_path =
        "replay_validation/exports/replay_validation_output.mp4";

    this->enable_visualization = true;

    this->enable_export = true;

    this->frame_counter = 0;
}


void ReplayValidationRuntime::initialize()
{
    std::cout << std::endl;

    std::cout
        << "=== Replay Validation Runtime ==="
        << std::endl;

    std::cout << std::endl;

    std::cout
        << "RGB Replay: "
        << rgb_video_path
        << std::endl;

    std::cout
        << "Depth Replay: "
        << depth_video_path
        << std::endl;

    std::cout << std::endl;
}


void ReplayValidationRuntime::run()
{
    std::cout
        << "Starting replay validation runtime..."
        << std::endl;

    std::cout << std::endl;

    // ------------------------------------------------------------
    // Future orchestration pipeline:
    //
    // ReplayLoader
    //     ->
    // SegmentationModule
    //     ->
    // OBB extraction
    //     ->
    // MultiObjectTracker
    //     ->
    // Overlay rendering
    //     ->
    // Telemetry rendering
    //     ->
    // Export pipeline
    // ------------------------------------------------------------

    std::cout
        << "Replay orchestration scaffold active."
        << std::endl;

    std::cout << std::endl;
}


void ReplayValidationRuntime::shutdown()
{
    std::cout << std::endl;

    std::cout
        << "Replay validation shutdown complete."
        << std::endl;

    std::cout
        << "Frames processed: "
        << frame_counter
        << std::endl;

    std::cout << std::endl;
}


void ReplayValidationRuntime::drawTelemetry()
{
    std::cout
        << "Telemetry rendering placeholder."
        << std::endl;
}


void ReplayValidationRuntime::drawTrackingOverlay()
{
    std::cout
        << "Tracking overlay placeholder."
        << std::endl;
}


void ReplayValidationRuntime::initializeExport()
{
    std::cout
        << "Initializing replay export pipeline..."
        << std::endl;
}


void ReplayValidationRuntime::finalizeExport()
{
    std::cout
        << "Finalizing replay export..."
        << std::endl;
}


int ReplayValidationRuntime::getFrameCount() const
{
    return frame_counter;
}