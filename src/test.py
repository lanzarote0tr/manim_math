from __future__ import annotations

import os
import subprocess
from typing import List

from manim import FadeIn, MathTex, Scene, Transform, config


def _fit_to_frame(mob: MathTex) -> None:
    max_width = config.frame_width * 0.9
    max_height = config.frame_height * 0.8
    if mob.width > max_width:
        mob.scale(max_width / mob.width)
    if mob.height > max_height:
        mob.scale(max_height / mob.height)
    mob.move_to([0, 0, 0])


class TestScene(Scene):
    def construct(self) -> None:
        expr = MathTex("1 + 2")
        _fit_to_frame(expr)
        self.play(FadeIn(expr))

        two = MathTex("3")
        _fit_to_frame(two)
        self.play(Transform(expr, two))

        three = MathTex("8")
        _fit_to_frame(three)
        self.play(Transform(expr, three))
        self.wait(1)


def main() -> None:
    cmd: List[str] = ["manim", "-pqh", os.path.abspath(__file__), "TestScene"]
    subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
