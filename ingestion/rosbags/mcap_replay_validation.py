from ingestion.rosbags.mcap_replay_loader import (
    MCAPReplayLoader
)


BAG_PATH = (

    "datasets/rosbags/rosbags/"
    "metric_depth_val_1779181947"
)


def main():

    print(
        "\n=== MCAP Replay Validation ===\n"
    )

    loader = MCAPReplayLoader(
        BAG_PATH
    )

    packet_count = 0

    while loader.has_next():

        packet = loader.get_next_packet()

        if packet is None:
            break

        print("=" * 60)

        print(
            f"Frame ID: "
            f"{packet.frame_id}"
        )

        print(
            f"Timestamp: "
            f"{packet.timestamp_ns}"
        )

        print()

        print(
            f"RGB Shape: "
            f"{packet.rgb_frame.shape}"
        )

        print(
            f"Depth Shape: "
            f"{packet.depth_frame.shape}"
        )

        packet_count += 1

        if packet_count >= 5:
            break

    loader.release()

    print(
        "\nValidation complete.\n"
    )


if __name__ == "__main__":
    main()