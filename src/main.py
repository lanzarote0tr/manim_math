from __future__ import annotations

import os
import re
import subprocess
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
        display_expr = os.environ.get("MANIM_RENDER_EXPR", expr)
        def_tex = os.environ.get("MANIM_DEF_TEX", "")
        def_line = os.environ.get("MANIM_DEF_LINE", "")
        anim_run_time = 1.2
        final_wait = 2.0

        title = Text("Step: 0", font="Noto Sans", weight="BOLD")
        title.scale(0.45).to_edge(UP, buff=0.1)

        def_mob = None
        if def_tex:
            def_mob = MathTex(def_tex)
            self._fit_to_frame(def_mob)
            def_mob.next_to(title, DOWN, buff=0.2)

        label = MathTex(to_latex(display_expr))
        self._fit_to_frame(label)
        if def_mob is not None:
            label.next_to(def_mob, DOWN, buff=0.35)

        if def_mob is not None:
            self.play(FadeIn(title), FadeIn(def_mob), FadeIn(label), run_time=anim_run_time)
        else:
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

        render_steps = steps
        phase1_len = 0
        if def_line:
            def_norm = re.sub(r"\s+", "", def_line)
            for idx, line in enumerate(steps):
                if re.sub(r"\s+", "", line) == def_norm:
                    phase1_len = idx
                    render_steps = steps[:idx] + steps[idx + 1 :]
                    break

        for i, line in enumerate(render_steps, start=1):
            if def_mob is not None and i == phase1_len + 1:
                self.play(FadeOut(def_mob), run_time=anim_run_time)
                def_mob = None
            new_title = Text(f"Step: {i}", font="Noto Sans", weight="BOLD")
            new_title.scale(0.45).to_edge(UP, buff=0.1)
            new_label = MathTex(to_latex(line))
            self._fit_to_frame(new_label)
            if def_mob is not None:
                new_label.next_to(def_mob, DOWN, buff=0.35)

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

    parts: List[str] = []
    for line in expr.splitlines():
        for chunk in line.split(";"):
            chunk = chunk.strip()
            if chunk:
                parts.append(chunk)

    def_re = re.compile(r"^\s*([A-Za-z])\s*\(\s*([A-Za-z])\s*\)\s*=\s*(.+)\s*$")
    def_lines: List[str] = []
    def_line_raw = ""
    display_expr = expr
    for part in parts:
        match = def_re.match(part)
        if match:
            def_line_raw = def_line_raw or f"{match.group(1)}({match.group(2)}) = {match.group(3).strip()}"
            def_lines.append(def_line_raw)
        else:
            display_expr = part

    def_tex = ""
    if def_lines:
        def_tex = r"\begin{aligned} " + r" \\ ".join(to_latex(line) for line in def_lines) + r" \end{aligned}"

    try:
        steps = solve(expr)
    except ParseError as exc:
        print(f"DEBUG_RENDER_PARSE_ERROR: {exc}")
        steps = []

    render_steps = steps
    phase1_len = 0
    if def_line_raw:
        def_norm = re.sub(r"\s+", "", def_line_raw)
        for idx, line in enumerate(steps):
            if re.sub(r"\s+", "", line) == def_norm:
                phase1_len = idx
                render_steps = steps[:idx] + steps[idx + 1 :]
                break

    print("DEBUG_RENDER_LIST:")
    if def_lines:
        print(f"Step 0 (top): {to_latex(def_lines[0])}")
        print(f"Step 0 (bottom): {to_latex(display_expr)}")
    else:
        print(f"Step 0: {to_latex(display_expr)}")
    for i, line in enumerate(render_steps, start=1):
        if def_lines and i == phase1_len + 1:
            print("Step merge: function definition removed")
        print(f"Step {i}: {to_latex(line)}")

    env = os.environ.copy()
    env[EXPR_ENV] = expr
    env["MANIM_RENDER_EXPR"] = display_expr
    env["MANIM_DEF_TEX"] = def_tex
    env["MANIM_DEF_LINE"] = def_line_raw

    cmd: List[str] = ["manim", "-pqh", os.path.abspath(__file__), "SolveScene"]
    subprocess.run(cmd, check=False, env=env)


if __name__ == "__main__":
    main()
