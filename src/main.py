from __future__ import annotations

import os
import re
import subprocess
import sys
from typing import List, Tuple

from manim import (
    DOWN,
    UP,
    FadeIn,
    FadeOut,
    MathTex,
    Scene,
    Text,
    Transform,
    TransformMatchingTex,
    config,
)

from solver import ParseError, solve


EXPR_ENV = "MANIM_EXPR"
DEFAULT_EXPR = "1 + 2x^2 + 3 + 3x + 7 = 0"
MAX_VISIBLE = 4


def _extract_parens(s: str, start: int) -> Tuple[str, int]:
    depth = 1
    i = start
    while i < len(s):
        if s[i] == "(":
            depth += 1
        elif s[i] == ")":
            depth -= 1
            if depth == 0:
                return s[start:i], i + 1
        i += 1
    return s[start:], len(s)


def _convert_roots(text: str) -> str:
    out = []
    i = 0
    while i < len(text):
        if text.startswith("sqrt(", i):
            inner, next_i = _extract_parens(text, i + 5)
            out.append(r"\sqrt{" + _convert_roots(inner) + "}")
            i = next_i
            continue
        if text.startswith("cbrt(", i):
            inner, next_i = _extract_parens(text, i + 5)
            out.append(r"\sqrt[3]{" + _convert_roots(inner) + "}")
            i = next_i
            continue
        out.append(text[i])
        i += 1
    return "".join(out)


def to_latex(text: str) -> str:
    converted = text.replace("+/-", r"\pm")
    converted = _convert_roots(converted)
    converted = re.sub(r"\bomega\b", r"\\omega", converted)
    converted = re.sub(r"\bDelta\b", r"\\Delta", converted)
    converted = re.sub(r"(?<!\\)([a-zA-Z])([0-9]+)", r"\1_{\2}", converted)
    return converted


class SolveScene(Scene):
    def construct(self) -> None:
        expr = os.environ.get(EXPR_ENV, DEFAULT_EXPR)
        anim_run_time = 1.2
        final_wait = 2.0

        title = Text("Step: 0", font="Noto Sans", weight="BOLD")
        title.scale(0.45).to_edge(UP, buff=0.1)

        label = MathTex(to_latex(expr))
        self._fit_to_frame(label)

        self.play(FadeIn(title), FadeIn(label), run_time=anim_run_time)

        try:
            steps = solve(expr)
        except ParseError as exc:
            error = Text(f"Parse error: {exc}", font="Noto Sans")
            error.scale(0.6).next_to(label, DOWN, buff=0.6)
            self.play(FadeIn(error), run_time=anim_run_time)
            self.wait(final_wait)
            return
        if not steps:
            note = Text("No steps.", font="Noto Sans")
            note.scale(0.6).next_to(label, DOWN, buff=0.6)
            self.play(FadeIn(note), run_time=anim_run_time)
            self.wait(final_wait)
            return

        for i, line in enumerate(steps, start=1):
            new_title = Text(f"Step: {i}", font="Noto Sans", weight="BOLD")
            new_title.scale(0.45).to_edge(UP, buff=0.1)
            new_label = MathTex(to_latex(line))
            self._fit_to_frame(new_label)

            self.play(
                Transform(title, new_title),
                Transform(label, new_label),
                run_time=anim_run_time,
            )

        self.wait(final_wait)

    def _fit_to_frame(self, mob: MathTex) -> None:
        max_width = config.frame_width * 0.9
        max_height = config.frame_height * 0.8
        if mob.width > max_width:
            mob.scale(max_width / mob.width)
        if mob.height > max_height:
            mob.scale(max_height / mob.height)
        mob.move_to([0, 0, 0])


def main() -> None:
    print("Enter expression or equation (use new lines or ';' for definitions, end with empty line):")
    lines: List[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if not line.strip():
            break
        lines.append(line)
    expr = "\n".join(lines).strip()
    if not expr:
        expr = DEFAULT_EXPR

    try:
        steps = solve(expr)
    except ParseError as exc:
        print(f"DEBUG_RENDER_PARSE_ERROR: {exc}")
        steps = []

    print("DEBUG_RENDER_LIST:")
    print(f"Step 0: {to_latex(expr)}")
    for i, line in enumerate(steps, start=1):
        print(f"Step {i}: {to_latex(line)}")

    env = os.environ.copy()
    env[EXPR_ENV] = expr

    cmd: List[str] = ["manim", "-pqh", os.path.abspath(__file__), "SolveScene"]
    subprocess.run(cmd, check=False, env=env)


if __name__ == "__main__":
    main()
