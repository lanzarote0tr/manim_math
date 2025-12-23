from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class Num:
    value: Fraction


@dataclass(frozen=True)
class Var:
    name: str


@dataclass(frozen=True)
class Neg:
    expr: "Expr"


@dataclass(frozen=True)
class BinOp:
    op: str
    left: "Expr"
    right: "Expr"


@dataclass(frozen=True)
class Ratio:
    left: "Expr"
    right: "Expr"


@dataclass(frozen=True)
class Group:
    expr: "Expr"


Expr = Num | Var | Neg | BinOp | Group


class ParseError(ValueError):
    pass


def tokenize(s: str) -> List[Tuple[str, str]]:
    tokens: List[Tuple[str, str]] = []
    i = 0
    while i < len(s):
        ch = s[i]
        if ch.isspace():
            i += 1
            continue
        if ch.isdigit():
            j = i
            while j < len(s) and s[j].isdigit():
                j += 1
            tokens.append(("NUMBER", s[i:j]))
            i = j
            continue
        if ch.isalpha():
            tokens.append(("IDENT", ch))
            i += 1
            continue
        if ch in "+-*/:=()":
            if ch == "(":
                tokens.append(("LPAREN", ch))
            elif ch == ")":
                tokens.append(("RPAREN", ch))
            else:
                tokens.append(("OP", ch))
            i += 1
            continue
        raise ParseError(f"Unexpected character: {ch}")
    return insert_implicit_mul(tokens)


def insert_implicit_mul(tokens: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    if not tokens:
        return tokens
    out: List[Tuple[str, str]] = [tokens[0]]
    for prev, curr in zip(tokens, tokens[1:]):
        if needs_implicit_mul(prev, curr):
            out.append(("OP", "*"))
        out.append(curr)
    return out


def needs_implicit_mul(prev: Tuple[str, str], curr: Tuple[str, str]) -> bool:
    prev_type, prev_val = prev
    curr_type, curr_val = curr
    if prev_type in ("NUMBER", "IDENT", "RPAREN"):
        if curr_type in ("NUMBER", "IDENT", "LPAREN"):
            if curr_val != "=" and prev_val != "=":
                return True
    return False


class Parser:
    def __init__(self, tokens: List[Tuple[str, str]]):
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> Optional[Tuple[str, str]]:
        if self.pos >= len(self.tokens):
            return None
        return self.tokens[self.pos]

    def consume(self, kind: str, value: Optional[str] = None) -> Tuple[str, str]:
        tok = self.peek()
        if tok is None:
            raise ParseError("Unexpected end of input")
        if tok[0] != kind or (value is not None and tok[1] != value):
            raise ParseError(f"Expected {kind} {value or ''}, got {tok}")
        self.pos += 1
        return tok

    def parse_expression(self) -> Expr:
        return self.parse_add_sub()

    def parse_add_sub(self) -> Expr:
        node = self.parse_mul_div()
        while True:
            tok = self.peek()
            if tok and tok[0] == "OP" and tok[1] in ("+", "-"):
                self.consume("OP")
                right = self.parse_mul_div()
                node = BinOp(tok[1], node, right)
            else:
                break
        return node

    def parse_mul_div(self) -> Expr:
        node = self.parse_unary()
        while True:
            tok = self.peek()
            if tok and tok[0] == "OP" and tok[1] in ("*", "/"):
                self.consume("OP")
                right = self.parse_unary()
                node = BinOp(tok[1], node, right)
            else:
                break
        return node

    def parse_unary(self) -> Expr:
        tok = self.peek()
        if tok and tok[0] == "OP" and tok[1] in ("+", "-"):
            self.consume("OP")
            expr = self.parse_unary()
            if tok[1] == "-":
                return Neg(expr)
            return expr
        return self.parse_primary()

    def parse_primary(self) -> Expr:
        tok = self.peek()
        if tok is None:
            raise ParseError("Unexpected end of input")
        if tok[0] == "NUMBER":
            self.consume("NUMBER")
            return Num(Fraction(int(tok[1]), 1))
        if tok[0] == "IDENT":
            self.consume("IDENT")
            return Var(tok[1])
        if tok[0] == "LPAREN":
            self.consume("LPAREN")
            expr = self.parse_expression()
            self.consume("RPAREN")
            return Group(expr)
        raise ParseError(f"Unexpected token: {tok}")


def parse_ratio(tokens: List[Tuple[str, str]]) -> Expr | Ratio:
    split_idx = find_top_level_op(tokens, ":")
    if split_idx is None:
        return parse_expr(tokens)
    left_tokens = tokens[:split_idx]
    right_tokens = tokens[split_idx + 1 :]
    left = parse_expr(left_tokens)
    right = parse_expr(right_tokens)
    return Ratio(left, right)


def parse_expr(tokens: List[Tuple[str, str]]) -> Expr:
    parser = Parser(tokens)
    expr = parser.parse_expression()
    if parser.peek() is not None:
        raise ParseError("Unexpected trailing input")
    return expr


def find_top_level_op(tokens: List[Tuple[str, str]], op: str) -> Optional[int]:
    depth = 0
    for i, (kind, val) in enumerate(tokens):
        if kind == "LPAREN":
            depth += 1
        elif kind == "RPAREN":
            depth -= 1
        elif depth == 0 and kind == "OP" and val == op:
            return i
    return None


def split_equation(tokens: List[Tuple[str, str]]) -> Optional[Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]]:
    idx = find_top_level_op(tokens, "=")
    if idx is None:
        return None
    return tokens[:idx], tokens[idx + 1 :]


def is_numeric(expr: Expr) -> bool:
    if isinstance(expr, Num):
        return True
    if isinstance(expr, Var):
        return False
    if isinstance(expr, Neg):
        return is_numeric(expr.expr)
    if isinstance(expr, BinOp):
        return is_numeric(expr.left) and is_numeric(expr.right)
    if isinstance(expr, Group):
        return is_numeric(expr.expr)
    return False


def collect_ops(expr: Expr) -> set[str]:
    if isinstance(expr, BinOp):
        return collect_ops(expr.left) | collect_ops(expr.right) | {expr.op}
    if isinstance(expr, Neg):
        return collect_ops(expr.expr)
    if isinstance(expr, Group):
        return collect_ops(expr.expr)
    return set()


def eval_numeric(expr: Expr) -> Fraction:
    if isinstance(expr, Num):
        return expr.value
    if isinstance(expr, Neg):
        return -eval_numeric(expr.expr)
    if isinstance(expr, BinOp):
        left = eval_numeric(expr.left)
        right = eval_numeric(expr.right)
        if expr.op == "+":
            return left + right
        if expr.op == "-":
            return left - right
        if expr.op == "*":
            return left * right
        if expr.op == "/":
            return left / right
    if isinstance(expr, Group):
        return eval_numeric(expr.expr)
    raise ValueError("Non-numeric expression")


def contains_group(expr: Expr) -> bool:
    if isinstance(expr, Group):
        return True
    if isinstance(expr, BinOp):
        return contains_group(expr.left) or contains_group(expr.right)
    if isinstance(expr, Neg):
        return contains_group(expr.expr)
    return False


def algebra_simplify(expr: Expr) -> Optional[Expr]:
    if isinstance(expr, BinOp):
        left = expr.left
        right = expr.right
        if expr.op == "+":
            if isinstance(left, Num) and left.value == 0:
                return right
            if isinstance(right, Num) and right.value == 0:
                return left
        if expr.op == "-":
            if isinstance(right, Num) and right.value == 0:
                return left
            if isinstance(left, Num) and left.value == 0:
                return Neg(right)
        if expr.op == "*":
            if isinstance(left, Num) and left.value == 0:
                return Num(Fraction(0))
            if isinstance(right, Num) and right.value == 0:
                return Num(Fraction(0))
            if isinstance(left, Num) and left.value == 1:
                return right
            if isinstance(right, Num) and right.value == 1:
                return left
        if expr.op == "/":
            if isinstance(left, Num) and left.value == 0:
                return Num(Fraction(0))
            if isinstance(right, Num) and right.value == 1:
                return left
    if isinstance(expr, Neg) and isinstance(expr.expr, Num):
        return Num(-expr.expr.value)
    if isinstance(expr, Group):
        if isinstance(expr.expr, (Num, Var, Neg, Group)):
            return expr.expr
    return None


def reduce_first_algebra(expr: Expr) -> Optional[Expr]:
    simplified = algebra_simplify(expr)
    if simplified is not None:
        return simplified
    if isinstance(expr, BinOp):
        new_left = reduce_first_algebra(expr.left)
        if new_left is not None:
            return BinOp(expr.op, new_left, expr.right)
        new_right = reduce_first_algebra(expr.right)
        if new_right is not None:
            return BinOp(expr.op, expr.left, new_right)
    if isinstance(expr, Neg):
        new_expr = reduce_first_algebra(expr.expr)
        if new_expr is not None:
            return Neg(new_expr)
    if isinstance(expr, Group):
        new_expr = reduce_first_algebra(expr.expr)
        if new_expr is not None:
            return Group(new_expr)
    return None


def numeric_collapse_once(expr: Expr) -> Optional[Expr]:
    if is_numeric(expr):
        if contains_group(expr):
            return None
        ops = collect_ops(expr)
        if ops <= {"+", "-"} or ops <= {"*", "/"}:
            return Num(eval_numeric(expr))
    return None


def reduce_first(expr: Expr, target_ops: set[str]) -> Optional[Expr]:
    if isinstance(expr, BinOp):
        new_left = reduce_first(expr.left, target_ops)
        if new_left is not None:
            return BinOp(expr.op, new_left, expr.right)
        new_right = reduce_first(expr.right, target_ops)
        if new_right is not None:
            return BinOp(expr.op, expr.left, new_right)
        if expr.op in target_ops and is_numeric(expr.left) and is_numeric(expr.right):
            return Num(eval_numeric(expr))
    if isinstance(expr, Neg):
        if is_numeric(expr.expr):
            return Num(-eval_numeric(expr.expr))
        new_expr = reduce_first(expr.expr, target_ops)
        if new_expr is not None:
            return Neg(new_expr)
    if isinstance(expr, Group):
        new_expr = reduce_first(expr.expr, target_ops)
        if new_expr is not None:
            return Group(new_expr)
    return None


def simplify_once(expr: Expr) -> Optional[Expr]:
    algebra = reduce_first_algebra(expr)
    if algebra is not None:
        return algebra
    collapsed = numeric_collapse_once(expr)
    if collapsed is not None:
        return collapsed
    for ops in ({"*", "/"}, {"+", "-"}):
        reduced = reduce_first(expr, ops)
        if reduced is not None:
            return reduced
    return None


def linear_form(expr: Expr) -> Optional[Tuple[Fraction, Fraction, Optional[str]]]:
    if isinstance(expr, Num):
        return Fraction(0), expr.value, None
    if isinstance(expr, Var):
        return Fraction(1), Fraction(0), expr.name
    if isinstance(expr, Neg):
        res = linear_form(expr.expr)
        if res is None:
            return None
        a, b, v = res
        return -a, -b, v
    if isinstance(expr, Group):
        return linear_form(expr.expr)
    if isinstance(expr, BinOp):
        if expr.op in ("+", "-"):
            left = linear_form(expr.left)
            right = linear_form(expr.right)
            if left is None or right is None:
                return None
            a1, b1, v1 = left
            a2, b2, v2 = right
            if v1 and v2 and v1 != v2:
                return None
            v = v1 or v2
            if expr.op == "+":
                return a1 + a2, b1 + b2, v
            return a1 - a2, b1 - b2, v
        if expr.op in ("*", "/"):
            left = linear_form(expr.left)
            right = linear_form(expr.right)
            if left is None or right is None:
                return None
            a1, b1, v1 = left
            a2, b2, v2 = right
            if expr.op == "*":
                if v1 and v2:
                    return None
                if v1:
                    return a1 * b2, b1 * b2, v1
                if v2:
                    return a2 * b1, b2 * b1, v2
                return Fraction(0), b1 * b2, None
            if v2:
                return None
            return a1 / b2, b1 / b2, v1
    return None


def format_number(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def precedence(expr: Expr) -> int:
    if isinstance(expr, Num) or isinstance(expr, Var):
        return 3
    if isinstance(expr, Neg):
        return 3
    if isinstance(expr, BinOp):
        if expr.op in ("*", "/"):
            return 2
        return 1
    if isinstance(expr, Group):
        return 3
    return 0


def format_expr(expr: Expr) -> str:
    if isinstance(expr, Num):
        return format_number(expr.value)
    if isinstance(expr, Var):
        return expr.name
    if isinstance(expr, Neg):
        inner = format_expr(expr.expr)
        if isinstance(expr.expr, (Num, Var)):
            return f"-{inner}"
        return f"-({inner})"
    if isinstance(expr, Group):
        inner = expr.expr
        if isinstance(inner, (Num, Var, Neg, Group)):
            return format_expr(inner)
        return f"({format_expr(inner)})"
    if isinstance(expr, BinOp):
        left = format_expr(expr.left)
        right = format_expr(expr.right)
        if precedence(expr.left) < precedence(expr):
            left = f"({left})"
        if precedence(expr.right) < precedence(expr) or (
            expr.op in ("-", "/") and precedence(expr.right) == precedence(expr)
        ):
            right = f"({right})"
        if expr.op == "*":
            if isinstance(expr.left, Num) and isinstance(expr.right, (Var, BinOp, Neg)):
                return f"{left}{right}"
            if isinstance(expr.left, Var) and isinstance(expr.right, BinOp):
                return f"{left}{right}"
        return f"{left} {expr.op} {right}"
    raise ValueError("Unknown expression")


def format_equation(left: Expr, right: Expr) -> str:
    return f"{format_expr(left)} = {format_expr(right)}"


def solve_expression_steps(expr: Expr, max_steps: int = 50) -> List[Expr]:
    steps = [expr]
    current = expr
    for _ in range(max_steps):
        new = simplify_once(current)
        if new is None or new == current:
            break
        if format_expr(new) != format_expr(current):
            current = new
            steps.append(current)
        else:
            current = new
    return steps


def simplify_equation_sides(left: Expr, right: Expr, max_steps: int = 10) -> List[Tuple[Expr, Expr]]:
    steps: List[Tuple[Expr, Expr]] = []
    cur_left, cur_right = left, right
    for _ in range(max_steps):
        new_left = simplify_once(cur_left)
        new_right = simplify_once(cur_right)
        if new_left is None and new_right is None:
            break
        if new_left is not None:
            cur_left = new_left
        if new_right is not None:
            cur_right = new_right
        steps.append((cur_left, cur_right))
    return steps


def solve_linear_equation_steps(left: Expr, right: Expr) -> Optional[List[Tuple[Expr, Expr]]]:
    left_form = linear_form(left)
    right_form = linear_form(right)
    if left_form is None or right_form is None:
        return None
    a1, b1, v1 = left_form
    a2, b2, v2 = right_form
    if v1 and v2 and v1 != v2:
        return None
    var = v1 or v2
    if var is None:
        return None
    a = a1 - a2
    b = b2 - b1
    if a == 0:
        return None
    steps: List[Tuple[Expr, Expr]] = []

    if is_numeric(left) and b2 == 0 and a2 != 0:
        left_steps = solve_expression_steps(left)
        cur_right = right
        for step in left_steps[1:]:
            steps.append((step, cur_right))
        simplified_right = simplify_once(cur_right)
        if simplified_right is not None and simplified_right != cur_right:
            cur_right = simplified_right
            steps.append((steps[-1][0], cur_right))
        if a2 != 1:
            steps.append((Var(var), BinOp("/", steps[-1][0], Num(a2))))
            simplified = simplify_once(steps[-1][1])
            if simplified is not None and simplified != steps[-1][1]:
                steps.append((Var(var), simplified))
        return steps

    left_var = BinOp("*", Num(a), Var(var)) if a != 1 else Var(var)
    right_expr: Expr
    if b != 0:
        if b > 0:
            right_expr = BinOp("-", Num(b2), Num(b1))
        else:
            right_expr = BinOp("+", Num(b2), Num(-b1))
    else:
        right_expr = Num(b)
    steps.append((left_var, right_expr))
    simplified = simplify_once(right_expr)
    if simplified is not None and simplified != right_expr:
        steps.append((left_var, simplified))
        right_expr = simplified

    if a != 1:
        new_right = BinOp("/", right_expr, Num(a))
        steps.append((Var(var), new_right))
        simplified = simplify_once(new_right)
        if simplified is not None and simplified != new_right:
            steps.append((Var(var), simplified))
    return steps


def solve_ratio_equation_steps(left: Ratio, right: Ratio) -> List[Tuple[Expr, Expr]]:
    cross_left = BinOp("*", left.right, right.left)
    cross_right = BinOp("*", left.left, right.right)
    steps = [(cross_left, cross_right)]
    return steps


def solve(input_str: str) -> List[str]:
    tokens = tokenize(input_str)
    split = split_equation(tokens)
    if split is None:
        expr = parse_ratio(tokens)
        if isinstance(expr, Ratio):
            expr = BinOp("/", expr.left, expr.right)
        expr_steps = solve_expression_steps(expr)
        lines: List[str] = []
        for step in expr_steps[1:]:
            rendered = format_expr(step)
            if not lines or lines[-1] != rendered:
                lines.append(rendered)
        return lines

    left_tokens, right_tokens = split
    left_expr = parse_ratio(left_tokens)
    right_expr = parse_ratio(right_tokens)

    steps: List[str] = []
    if isinstance(left_expr, Ratio) and isinstance(right_expr, Ratio):
        ratio_steps = solve_ratio_equation_steps(left_expr, right_expr)
        for l, r in ratio_steps:
            rendered = format_equation(l, r)
            if not steps or steps[-1] != rendered:
                steps.append(rendered)
        left_expr, right_expr = ratio_steps[-1]
        simplified_steps = simplify_equation_sides(left_expr, right_expr)
        for l, r in simplified_steps:
            rendered = format_equation(l, r)
            if not steps or steps[-1] != rendered:
                steps.append(rendered)
        if simplified_steps:
            left_expr, right_expr = simplified_steps[-1]

    linear_steps = solve_linear_equation_steps(left_expr, right_expr)
    if linear_steps is not None:
        if linear_steps:
            last_left, last_right = linear_steps[-1]
            if is_numeric(last_left) and isinstance(last_right, Var):
                linear_steps.append((last_right, last_left))
        for l, r in linear_steps:
            rendered = format_equation(l, r)
            if not steps or steps[-1] != rendered:
                steps.append(rendered)
        return steps

    left_steps = solve_expression_steps(left_expr)
    right_steps = solve_expression_steps(right_expr)
    for l, r in zip(left_steps[1:], right_steps[1:]):
        rendered = format_equation(l, r)
        if not steps or steps[-1] != rendered:
            steps.append(rendered)
    return steps


def main() -> None:
    s = input("Enter expression or equation: ").strip()
    try:
        steps = solve(s)
    except ParseError as exc:
        print(f"Parse error: {exc}")
        return

    if not steps:
        print("No steps.")
        return

    for i, line in enumerate(steps, start=1):
        print(f"Step {i}: {line}")


if __name__ == "__main__":
    main()
