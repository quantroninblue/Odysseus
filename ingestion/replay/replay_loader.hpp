// replay_loader.hpp

#ifndef REPLAY_LOADER_HPP
#define REPLAY_LOADER_HPP

#include <string>
#include <opencv2/opencv.hpp>


struct ReplayFramePacket
{
    int frame_id;

    double timestamp;

    cv::Mat rgb_frame;

    cv::Mat depth_frame;
};


class ReplayLoader
{
public:

    ReplayLoader(
        const std::string& rgb_video_path,
        const std::string& depth_video_path = "",
        bool loop = false
    );

    bool hasNext() const;

    ReplayFramePacket getNextPacket();

    void reset();

    void release();

private:

    cv::VideoCapture rgb_cap_;

    cv::VideoCapture depth_cap_;

    bool has_depth_;

    bool loop_;

    int frame_index_;

    int total_frames_;

    double fps_;
};

#endif