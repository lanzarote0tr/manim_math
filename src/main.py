import sympy as sp
from sympy.parsing.sympy_parser import parse_expr

x = sp.Symbol("x")

def parse_equation(s: str):
    if "=" not in s:
        raise ValueError("Equation must contain '='")
    left, right = s.split("=")
    return (
        parse_expr(left.strip(), evaluate=False),
        parse_expr(right.strip(), evaluate=False),
    )

def print_step(n, left, right):
    print(f"Step {n}: {sp.latex(left)} = {sp.latex(right)}")

def move_constant(left, right):
    # assumes left = ax + b
    a, b = left.as_ordered_terms()
    return a, right - b

def divide_coefficient(left, right):
    coeff = left.as_coeff_Mul()[0]
    var = left / coeff
    return var, right / coeff

def solve_linear_equation(eq_str):
    left, right = parse_equation(eq_str)
    step = 1

    print_step(step, left, right)

    # Step 2: move constant
    step += 1
    left, right = move_constant(left, right)
    print_step(step, left, right)

    # Step 3: simplify right
    step += 1
    right = sp.simplify(right)
    print_step(step, left, right)

    # Step 4: divide coefficient
    step += 1
    left, right = divide_coefficient(left, right)
    print_step(step, left, right)

if __name__ == "__main__":
    eq = input("Enter equation (e.g. 2*x+3=11): ")
    solve_linear_equation(eq)


'''
from manim import *
import re


class Main(Scene):
    def construct(self):
        s = "3+4"
        result = re.findall(r'\d+|[+\-*/]', s)
        print(result)
        
        expr = MathTex(s.split())
        result = MathTex("7")
        
        expr.move_to(ORIGIN)
        result.move_to(ORIGIN)
        
        self.play(Write(expr))
        self.wait(0.5)

        self.play(
            TransformMatchingTex(
                expr,
                result,
                transform_mismatches=True
            )
        )
        self.wait(1)
'''

