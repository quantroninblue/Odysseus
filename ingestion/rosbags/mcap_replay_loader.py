from pathlib import Path

import numpy as np

from rosbags.highlevel import AnyReader


class MCAPReplayPacket:

    def __init__(

        self,

        frame_id,

        timestamp_ns,

        rgb_frame,

        depth_frame
    ):

        self.frame_id = frame_id

        self.timestamp_ns = (
            timestamp_ns
        )

        self.rgb_frame = (
            rgb_frame
        )

        self.depth_frame = (
            depth_frame
        )


class MCAPReplayLoader:

    def __init__(

        self,

        bag_path
    ):

        self.bag_path = Path(
            bag_path
        )

        self.reader = None

        self.messages = []

        self.current_index = 0

        self.frame_counter = 0

        # ----------------------------------------------------
        # Open reader
        # ----------------------------------------------------

        self.reader = AnyReader(
            [self.bag_path]
        )

        self.reader.open()

        print(
            "\n=== MCAP Replay Loader Initialized ===\n"
        )

        print(
            f"Bag Path:\n"
            f"{self.bag_path}\n"
        )

        # ----------------------------------------------------
        # Cache all messages
        # ----------------------------------------------------

        self.messages = list(
            self.reader.messages()
        )

        print(
            f"Total Messages: "
            f"{len(self.messages)}\n"
        )

    # --------------------------------------------------------
    # Runtime state
    # --------------------------------------------------------

    def has_next(self):

        return (
            self.current_index <
            len(self.messages)
        )

    # --------------------------------------------------------
    # Packet extraction
    # --------------------------------------------------------

    def get_next_packet(self):

        rgb_frame = None

        depth_frame = None

        rgb_timestamp = None

        depth_timestamp = None

        # ----------------------------------------------------
        # Read until both frames acquired
        # ----------------------------------------------------

        while self.has_next():

            (
                connection,
                timestamp,
                rawdata
            ) = self.messages[
                self.current_index
            ]

            self.current_index += 1

            msg = self.reader.deserialize(

                rawdata,

                connection.msgtype
            )

            # ------------------------------------------------
            # RGB
            # ------------------------------------------------

            if (
                connection.topic ==
                "/vctr/rgb_raw"
            ):

                rgb = np.frombuffer(

                    msg.data,

                    dtype=np.uint8
                )

                rgb = rgb.reshape(

                    msg.height,

                    msg.width,

                    3
                )

                rgb_frame = rgb

                rgb_timestamp = timestamp

            # ------------------------------------------------
            # DEPTH
            # ------------------------------------------------

            elif (
                connection.topic ==
                "/vctr/depth_raw"
            ):

                depth = np.frombuffer(

                    msg.data,

                    dtype=np.uint16
                )

                depth = depth.reshape(

                    msg.height,

                    msg.width
                )

                depth_frame = depth

                depth_timestamp = timestamp

            # ------------------------------------------------
            # Packet ready
            # ------------------------------------------------

            if (

                rgb_frame is not None and

                depth_frame is not None
            ):

                packet = MCAPReplayPacket(

                    frame_id=self.frame_counter,

                    timestamp_ns=timestamp,

                    rgb_frame=rgb_frame,

                    depth_frame=depth_frame
                )

                self.frame_counter += 1

                return packet

        return None

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    def release(self):

        if self.reader is not None:

            self.reader.close()

        print(
            "\nMCAP replay loader released.\n"
        )