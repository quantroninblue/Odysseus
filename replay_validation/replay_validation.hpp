#pragma once

/*
replay_validation.hpp

Replay-driven modular integration validation runtime.

Purpose:
- Orchestrate subsystem interoperability
- Validate integrated perception runtime
- Execute replay-driven perception evaluation

This module DOES NOT implement:
- segmentation
- geometry
- tracking
- visualization internals

It ONLY orchestrates subsystem interaction.
*/

#include <string>


class ReplayValidationRuntime
{
private:

    // ------------------------------------------------------------
    // Replay session
    // ------------------------------------------------------------

    std::string rgb_video_path;

    std::string depth_video_path;

    // ------------------------------------------------------------
    // Export configuration
    // ------------------------------------------------------------

    std::string export_directory;

    std::string export_video_path;

    // ------------------------------------------------------------
    // Runtime configuration
    // ------------------------------------------------------------

    bool enable_visualization;

    bool enable_export;

    // ------------------------------------------------------------
    // Runtime state
    // ------------------------------------------------------------

    int frame_counter;

public:

    ReplayValidationRuntime(

        const std::string& rgb_video_path,

        const std::string& depth_video_path
    );

    // ------------------------------------------------------------
    // Runtime execution
    // ------------------------------------------------------------

    void initialize();

    void run();

    void shutdown();

    // ------------------------------------------------------------
    // Visualization helpers
    // ------------------------------------------------------------

    void drawTelemetry();

    void drawTrackingOverlay();

    // ------------------------------------------------------------
    // Export helpers
    // ------------------------------------------------------------

    void initializeExport();

    void finalizeExport();

    // ------------------------------------------------------------
    // Runtime info
    // ------------------------------------------------------------

    int getFrameCount() const;
};