import cv2

from replay_loader import ReplayLoader


def main():

    print("\n=== Replay Validation ===\n")

    rgb_video = (
        "../../replay_sessions/"
        "ishaan_15.05.2026_singlebox/"
        "recordings/rgb_recording.mp4"
    )

    depth_video = (
        "../../replay_sessions/"
        "ishaan_15.05.2026_singlebox/"
        "recordings/depth_recording.mp4"
    )

    loader = ReplayLoader(
        rgb_video_path=rgb_video,
        depth_video_path=depth_video,
        loop=False
    )

    frame_counter = 0

    while loader.has_next():

        packet = loader.get_next_packet()

        if packet is None:
            break

        print(
            f"\nFrame ID: {packet.frame_id}"
        )

        print(
            f"RGB Shape: "
            f"{packet.rgb_frame.shape}"
        )

        if packet.depth_frame is not None:

            print(
                f"Depth Shape: "
                f"{packet.depth_frame.shape}"
            )

        # --------------------------------------------------------
        # RGB visualization
        # --------------------------------------------------------

        rgb_vis = packet.rgb_frame.copy()

        cv2.putText(
            rgb_vis,
            f"Replay Frame {packet.frame_id}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )

        cv2.imshow(
            "RGB Replay",
            rgb_vis
        )

        # --------------------------------------------------------
        # Depth visualization
        # --------------------------------------------------------

        if packet.depth_frame is not None:

            cv2.imshow(
                "Depth Replay",
                packet.depth_frame
            )

        key = cv2.waitKey(30)

        if key == 27:
            break

        frame_counter += 1

    loader.release()

    cv2.destroyAllWindows()

    print("\nReplay validation complete.")
    print(f"Frames processed: {frame_counter}")


if __name__ == "__main__":
    main()