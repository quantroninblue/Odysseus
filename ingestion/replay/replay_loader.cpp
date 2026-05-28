// replay_loader.cpp

#include "replay_loader.hpp"

#include <iostream>
#include <stdexcept>


ReplayLoader::ReplayLoader(
    const std::string& rgb_video_path,
    const std::string& depth_video_path,
    bool loop
)
{
    loop_ = loop;

    frame_index_ = 0;

    rgb_cap_.open(rgb_video_path);

    if (!rgb_cap_.isOpened())
    {
        throw std::runtime_error(
            "Failed opening RGB replay video."
        );
    }

    has_depth_ = false;

    if (!depth_video_path.empty())
    {
        depth_cap_.open(depth_video_path);

        if (!depth_cap_.isOpened())
        {
            throw std::runtime_error(
                "Failed opening depth replay video."
            );
        }

        has_depth_ = true;
    }

    total_frames_ = static_cast<int>(
        rgb_cap_.get(cv::CAP_PROP_FRAME_COUNT)
    );

    fps_ = rgb_cap_.get(
        cv::CAP_PROP_FPS
    );

    std::cout << "\n=== Replay Loader Initialized ===\n";

    std::cout << "Frames: "
              << total_frames_
              << std::endl;

    std::cout << "FPS: "
              << fps_
              << std::endl;
}


bool ReplayLoader::hasNext() const
{
    return frame_index_ < total_frames_;
}


ReplayFramePacket ReplayLoader::getNextPacket()
{
    ReplayFramePacket packet;

    cv::Mat rgb_frame;

    bool rgb_ok =
        rgb_cap_.read(rgb_frame);

    if (!rgb_ok)
    {
        return packet;
    }

    packet.rgb_frame = rgb_frame;

    if (has_depth_)
    {
        cv::Mat depth_frame;

        bool depth_ok =
            depth_cap_.read(depth_frame);

        if (depth_ok)
        {
            packet.depth_frame = depth_frame;
        }
    }

    packet.frame_id = frame_index_;

    packet.timestamp =
        static_cast<double>(frame_index_) / fps_;

    frame_index_++;

    return packet;
}


void ReplayLoader::reset()
{
    rgb_cap_.set(
        cv::CAP_PROP_POS_FRAMES,
        0
    );

    if (has_depth_)
    {
        depth_cap_.set(
            cv::CAP_PROP_POS_FRAMES,
            0
        );
    }

    frame_index_ = 0;
}


void ReplayLoader::release()
{
    rgb_cap_.release();

    if (has_depth_)
    {
        depth_cap_.release();
    }
}