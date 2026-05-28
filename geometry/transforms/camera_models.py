from dataclasses import dataclass


@dataclass
class CameraIntrinsics:

    fx: float
    fy: float

    cx: float
    cy: float

    width: int
    height: int


    def scaled_to_resolution(

        self,

        new_width: int,
        new_height: int
    ):

        scale_x = (
            new_width / self.width
        )

        scale_y = (
            new_height / self.height
        )

        return CameraIntrinsics(

            fx=self.fx * scale_x,
            fy=self.fy * scale_y,

            cx=self.cx * scale_x,
            cy=self.cy * scale_y,

            width=new_width,
            height=new_height
        )


    def print_summary(self):

        print("\n=== Camera Intrinsics ===\n")

        print(
            f"Resolution: "
            f"{self.width} x {self.height}"
        )

        print(
            f"fx: {self.fx:.3f}"
        )

        print(
            f"fy: {self.fy:.3f}"
        )

        print(
            f"cx: {self.cx:.3f}"
        )

        print(
            f"cy: {self.cy:.3f}"
        )

        print()