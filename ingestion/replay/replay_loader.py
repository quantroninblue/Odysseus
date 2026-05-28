import cv2
import time

from ingestion.replay.replay_packet import ReplayFramePacket


class ReplayLoader:

    def __init__(
        self,
        rgb_video_path,
        depth_video_path=None,
        loop=False
    ):

        self.rgb_video_path = rgb_video_path
        self.depth_video_path = depth_video_path

        self.loop = loop

        self.frame_index = 0

        # --------------------------------------------------------
        # Open video streams
        # --------------------------------------------------------

        self.rgb_cap = cv2.VideoCapture(
            self.rgb_video_path
        )

        if not self.rgb_cap.isOpened():

            raise RuntimeError(
                f"Failed opening RGB video:\n"
                f"{self.rgb_video_path}"
            )

        self.depth_cap = None

        if self.depth_video_path is not None:

            self.depth_cap = cv2.VideoCapture(
                self.depth_video_path
            )

            if not self.depth_cap.isOpened():

                raise RuntimeError(
                    f"Failed opening depth video:\n"
                    f"{self.depth_video_path}"
                )

        self.total_frames = int(
            self.rgb_cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        self.fps = self.rgb_cap.get(
            cv2.CAP_PROP_FPS
        )

        print("\n=== Replay Loader Initialized ===\n")

        print(f"RGB Video:")
        print(self.rgb_video_path)

        if self.depth_video_path is not None:

            print(f"\nDepth Video:")
            print(self.depth_video_path)

        print(f"\nTotal Frames: {self.total_frames}")
        print(f"FPS: {self.fps:.2f}")

        print()

    def has_next(self):

        if self.loop:
            return True

        return (
            self.frame_index <
            self.total_frames
        )

    def reset(self):

        self.rgb_cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            0
        )

        if self.depth_cap is not None:

            self.depth_cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                0
            )

        self.frame_index = 0

    def get_next_packet(self):

        if not self.has_next():

            if self.loop:

                self.reset()

            else:
                return None

        # --------------------------------------------------------
        # RGB frame
        # --------------------------------------------------------

        rgb_ok, rgb_frame = self.rgb_cap.read()

        if not rgb_ok:

            return None

        # --------------------------------------------------------
        # Depth frame
        # --------------------------------------------------------

        depth_frame = None

        if self.depth_cap is not None:

            depth_ok, depth_frame = self.depth_cap.read()

            if not depth_ok:

                depth_frame = None

        # --------------------------------------------------------
        # Build replay packet
        # --------------------------------------------------------

        packet = ReplayFramePacket(

            frame_id=self.frame_index,

            timestamp=time.time(),

            rgb_frame=rgb_frame,

            depth_frame=depth_frame,

            rgb_path=self.rgb_video_path,

            depth_path=self.depth_video_path
        )

        self.frame_index += 1

        return packet

    def release(self):

        self.rgb_cap.release()

        if self.depth_cap is not None:

            self.depth_cap.release()