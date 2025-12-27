from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import isqrt
import re
from typing import List, Optional, Tuple

Token = Tuple[str, str]
Tokens = List[Token]
Poly = dict[int, Fraction]


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
class Pow:
    base: "Expr"
    exponent: "Expr"


@dataclass(frozen=True)
class Imag:
    pass


@dataclass(frozen=True)
class Ratio:
    left: "Expr"
    right: "Expr"


@dataclass(frozen=True)
class Group:
    expr: "Expr"


Expr = Num | Var | Neg | BinOp | Pow | Group | Imag


class ParseError(ValueError):
    pass


FUNC_DEF_RE = re.compile(r"^\s*([A-Za-z])\s*\(\s*([A-Za-z])\s*\)\s*=\s*(.+)\s*$")


def _linear_term(expr: Expr) -> Optional[Tuple[Fraction, Var]]:
    if isinstance(expr, Var):
        return Fraction(1), expr
    if isinstance(expr, Neg):
        inner = _linear_term(expr.expr)
        if inner is None:
            return None
        coef, var = inner
        return -coef, var
    if isinstance(expr, BinOp) and expr.op == "*":
        if isinstance(expr.left, Num) and isinstance(expr.right, Var):
            return expr.left.value, expr.right
        if isinstance(expr.right, Num) and isinstance(expr.left, Var):
            return expr.right.value, expr.left
    return None


def _build_linear_term(coef: Fraction, var: Var) -> Expr:
    if coef == 0:
        return Num(Fraction(0))
    if coef == 1:
        return var
    if coef == -1:
        return Neg(var)
    return BinOp("*", Num(coef), var)


ADD_OPS = {"+", "-"}
MUL_OPS = {"*", "/"}


def tokenize(s: str) -> Tokens:
    tokens: Tokens = []
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
        if ch in "+-*/:=()^":
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


def insert_implicit_mul(tokens: Tokens) -> Tokens:
    if not tokens:
        return tokens
    out: Tokens = [tokens[0]]
    for prev, curr in zip(tokens, tokens[1:]):
        if needs_implicit_mul(prev, curr):
            out.append(("OP", "*"))
        out.append(curr)
    return out


def needs_implicit_mul(prev: Token, curr: Token) -> bool:
    prev_type, prev_val = prev
    curr_type, curr_val = curr
    if prev_type == "NUMBER":
        if curr_type in ("IDENT", "LPAREN"):
            return curr_val != "=" and prev_val != "="
        return False
    if prev_type in ("IDENT", "RPAREN"):
        if curr_type in ("NUMBER", "IDENT", "LPAREN"):
            return curr_val != "=" and prev_val != "="
    return False


class Parser:
    def __init__(self, tokens: Tokens):
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> Optional[Token]:
        if self.pos >= len(self.tokens):
            return None
        return self.tokens[self.pos]

    def consume(self, kind: str, value: Optional[str] = None) -> Token:
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
            if tok and tok[0] == "OP" and tok[1] in ADD_OPS:
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
            if tok and tok[0] == "OP" and tok[1] in MUL_OPS:
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
        return self.parse_power()

    def parse_power(self) -> Expr:
        node = self.parse_primary()
        tok = self.peek()
        if tok and tok[0] == "OP" and tok[1] == "^":
            self.consume("OP")
            exponent = self.parse_unary()
            node = Pow(node, exponent)
        return node

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


def parse_ratio(tokens: Tokens) -> Expr | Ratio:
    split_idx = find_top_level_op(tokens, ":")
    if split_idx is None:
        return parse_expr(tokens)
    left_tokens = tokens[:split_idx]
    right_tokens = tokens[split_idx + 1 :]
    left = parse_expr(left_tokens)
    right = parse_expr(right_tokens)
    return Ratio(left, right)


def parse_expr(tokens: Tokens) -> Expr:
    parser = Parser(tokens)
    expr = parser.parse_expression()
    if parser.peek() is not None:
        raise ParseError("Unexpected trailing input")
    return expr


def find_top_level_op(tokens: Tokens, op: str) -> Optional[int]:
    depth = 0
    for i, (kind, val) in enumerate(tokens):
        if kind == "LPAREN":
            depth += 1
        elif kind == "RPAREN":
            depth -= 1
        elif depth == 0 and kind == "OP" and val == op:
            return i
    return None


def split_equation(tokens: Tokens) -> Optional[Tuple[Tokens, Tokens]]:
    idx = find_top_level_op(tokens, "=")
    if idx is None:
        return None
    return tokens[:idx], tokens[idx + 1 :]


def is_numeric(expr: Expr) -> bool:
    if isinstance(expr, Num):
        return True
    if isinstance(expr, (Imag, Var)):
        return False
    if isinstance(expr, Neg):
        return is_numeric(expr.expr)
    if isinstance(expr, BinOp):
        return is_numeric(expr.left) and is_numeric(expr.right)
    if isinstance(expr, Pow):
        return is_numeric(expr.base) and is_numeric(expr.exponent)
    if isinstance(expr, Group):
        return is_numeric(expr.expr)
    return False


def collect_ops(expr: Expr) -> set[str]:
    if isinstance(expr, BinOp):
        return collect_ops(expr.left) | collect_ops(expr.right) | {expr.op}
    if isinstance(expr, Pow):
        return collect_ops(expr.base) | collect_ops(expr.exponent) | {"^"}
    if isinstance(expr, Neg):
        return collect_ops(expr.expr)
    if isinstance(expr, Group):
        return collect_ops(expr.expr)
    return set()


def eval_numeric(expr: Expr) -> Fraction:
    if isinstance(expr, Num):
        return expr.value
    if isinstance(expr, Imag):
        raise ValueError("Non-numeric expression")
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
    if isinstance(expr, Pow):
        base = eval_numeric(expr.base)
        exponent = eval_numeric(expr.exponent)
        if exponent.denominator != 1:
            raise ValueError("Non-integer exponent")
        return base ** int(exponent)
    if isinstance(expr, Group):
        return eval_numeric(expr.expr)
    raise ValueError("Non-numeric expression")


def contains_group(expr: Expr) -> bool:
    if isinstance(expr, Group):
        return True
    if isinstance(expr, BinOp):
        return contains_group(expr.left) or contains_group(expr.right)
    if isinstance(expr, Pow):
        return contains_group(expr.base) or contains_group(expr.exponent)
    if isinstance(expr, Neg):
        return contains_group(expr.expr)
    return False


def algebra_simplify(expr: Expr) -> Optional[Expr]:
    if isinstance(expr, Pow):
        if isinstance(expr.exponent, Num):
            if expr.exponent.value == 0:
                return Num(Fraction(1))
            if expr.exponent.value == 1:
                return expr.base
        if isinstance(expr.base, Num) and expr.base.value == 1:
            return Num(Fraction(1))
    if isinstance(expr, BinOp):
        left = expr.left
        right = expr.right
        if expr.op == "+":
            left_term = _linear_term(left)
            right_term = _linear_term(right)
            if left_term and right_term and left_term[1] == right_term[1]:
                return _build_linear_term(left_term[0] + right_term[0], left_term[1])
            if isinstance(left, Num) and left.value == 0:
                return right
            if isinstance(right, Num) and right.value == 0:
                return left
        if expr.op == "-":
            left_term = _linear_term(left)
            right_term = _linear_term(right)
            if left_term and right_term and left_term[1] == right_term[1]:
                return _build_linear_term(left_term[0] - right_term[0], left_term[1])
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
    if isinstance(expr, Pow):
        new_base = reduce_first_algebra(expr.base)
        if new_base is not None:
            return Pow(new_base, expr.exponent)
        new_exp = reduce_first_algebra(expr.exponent)
        if new_exp is not None:
            return Pow(expr.base, new_exp)
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
        if ops <= ADD_OPS or ops <= MUL_OPS:
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
    if isinstance(expr, Pow):
        if is_numeric(expr.base) and is_numeric(expr.exponent):
            try:
                return Num(eval_numeric(expr))
            except ValueError:
                return None
        new_base = reduce_first(expr.base, target_ops)
        if new_base is not None:
            return Pow(new_base, expr.exponent)
        new_exp = reduce_first(expr.exponent, target_ops)
        if new_exp is not None:
            return Pow(expr.base, new_exp)
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
    for ops in (MUL_OPS, ADD_OPS):
        reduced = reduce_first(expr, ops)
        if reduced is not None:
            return reduced
    return None


def find_vars(expr: Expr) -> set[str]:
    if isinstance(expr, Var):
        return {expr.name}
    if isinstance(expr, (Num, Imag)):
        return set()
    if isinstance(expr, Neg):
        return find_vars(expr.expr)
    if isinstance(expr, Group):
        return find_vars(expr.expr)
    if isinstance(expr, BinOp):
        return find_vars(expr.left) | find_vars(expr.right)
    if isinstance(expr, Pow):
        return find_vars(expr.base) | find_vars(expr.exponent)
    return set()


def poly_add(p: Poly, q: Poly) -> Poly:
    out = dict(p)
    for deg, coef in q.items():
        out[deg] = out.get(deg, Fraction(0)) + coef
        if out[deg] == 0:
            del out[deg]
    return out


def poly_mul(p: Poly, q: Poly) -> Poly:
    out: Poly = {}
    for d1, c1 in p.items():
        for d2, c2 in q.items():
            d = d1 + d2
            out[d] = out.get(d, Fraction(0)) + c1 * c2
            if out[d] == 0:
                del out[d]
    return out


def poly_from_expr(expr: Expr, var: str) -> Optional[Poly]:
    if isinstance(expr, Num):
        return {0: expr.value}
    if isinstance(expr, Var):
        if expr.name == var:
            return {1: Fraction(1)}
        return None
    if isinstance(expr, Imag):
        return None
    if isinstance(expr, Group):
        return poly_from_expr(expr.expr, var)
    if isinstance(expr, Neg):
        inner = poly_from_expr(expr.expr, var)
        if inner is None:
            return None
        return {deg: -coef for deg, coef in inner.items()}
    if isinstance(expr, BinOp):
        left = poly_from_expr(expr.left, var)
        right = poly_from_expr(expr.right, var)
        if expr.op in ADD_OPS:
            if left is None or right is None:
                return None
            if expr.op == "-":
                right = {deg: -coef for deg, coef in right.items()}
            return poly_add(left, right)
        if expr.op == "*":
            if left is None or right is None:
                return None
            return poly_mul(left, right)
        if expr.op == "/":
            if left is None or right is None:
                return None
            if right.keys() == {0}:
                divisor = right[0]
                if divisor == 0:
                    return None
                return {deg: coef / divisor for deg, coef in left.items()}
            return None
        return None
    if isinstance(expr, Pow):
        if not isinstance(expr.exponent, Num):
            return None
        if expr.exponent.value.denominator != 1:
            return None
        exp = int(expr.exponent.value)
        if exp < 0:
            return None
        base = poly_from_expr(expr.base, var)
        if base is None:
            return None
        result: Poly = {0: Fraction(1)}
        for _ in range(exp):
            result = poly_mul(result, base)
        return result
    return None


def poly_degree(poly: Poly) -> int:
    if not poly:
        return 0
    return max(poly.keys())


def add_constant(expr: Expr, value: Fraction) -> Expr:
    if value >= 0:
        return BinOp("+", expr, Num(value))
    return BinOp("-", expr, Num(-value))


def sqrt_fraction(value: Fraction) -> Expr:
    if value == 0:
        return Num(Fraction(0))
    if value > 0:
        num = value.numerator
        den = value.denominator
        num_root = isqrt(num)
        den_root = isqrt(den)
        if num_root * num_root == num and den_root * den_root == den:
            return Num(Fraction(num_root, den_root))
        if den_root * den_root == den:
            if den_root == 1:
                return Pow(Num(num), Num(Fraction(1, 2)))
            return BinOp("/", Pow(Num(num), Num(Fraction(1, 2))), Num(den_root))
        return Pow(Num(value), Num(Fraction(1, 2)))
    pos = -value
    num = pos.numerator
    den = pos.denominator
    den_root = isqrt(den)
    if den_root * den_root == den:
        if den_root == 1:
            return BinOp("*", Imag(), Pow(Num(num), Num(Fraction(1, 2))))
        return BinOp("*", Imag(), BinOp("/", Pow(Num(num), Num(Fraction(1, 2))), Num(den_root)))
    return BinOp("*", Imag(), Pow(Num(pos), Num(Fraction(1, 2))))


def build_poly_expr(coeffs: Poly, var: str) -> Expr:
    if not coeffs:
        return Num(Fraction(0))
    terms: List[Expr] = []
    for deg in sorted(coeffs.keys(), reverse=True):
        coef = coeffs[deg]
        if coef == 0:
            continue
        if deg == 0:
            term = Num(coef)
        elif deg == 1:
            if coef == 1:
                term = Var(var)
            elif coef == -1:
                term = Neg(Var(var))
            else:
                term = BinOp("*", Num(coef), Var(var))
        else:
            power = Pow(Var(var), Num(Fraction(deg, 1)))
            if coef == 1:
                term = power
            elif coef == -1:
                term = Neg(power)
            else:
                term = BinOp("*", Num(coef), power)
        terms.append(term)
    if not terms:
        return Num(Fraction(0))
    expr = terms[0]
    for term in terms[1:]:
        if isinstance(term, Neg):
            expr = BinOp("-", expr, term.expr)
        else:
            expr = BinOp("+", expr, term)
    return expr


def quadratic_steps(a: Fraction, b: Fraction, c: Fraction, var: str) -> List[str]:
    x = Var(var)
    steps: List[str] = []

    standard = build_poly_expr({2: a, 1: b, 0: c}, var)
    steps.append(format_equation(standard, Num(Fraction(0))))

    if a != 1:
        steps.append(
            format_equation(
                BinOp(
                    "+",
                    BinOp("+", Pow(x, Num(Fraction(2))), BinOp("*", Num(b / a), x)),
                    Num(c / a),
                ),
                Num(Fraction(0)),
            )
        )

    steps.append(
        format_equation(
            BinOp("+", Pow(x, Num(Fraction(2))), BinOp("*", Num(b / a), x)),
            Num(-c / a),
        )
    )

    half_b_over_a = b / (2 * a)
    square_term = Num(half_b_over_a * half_b_over_a)
    left_complete = BinOp(
        "+",
        BinOp("+", Pow(x, Num(Fraction(2))), BinOp("*", Num(b / a), x)),
        square_term,
    )
    right_complete = BinOp("+", Num(-c / a), square_term)
    steps.append(format_equation(left_complete, right_complete))

    disc = b * b - 4 * a * c
    left_sq = Pow(add_constant(x, half_b_over_a), Num(Fraction(2)))
    right_sq = Num(disc / (4 * a * a))
    steps.append(format_equation(left_sq, right_sq))

    sqrt_right = sqrt_fraction(disc / (4 * a * a))
    steps.append(f"{format_expr(add_constant(x, half_b_over_a))} = +/- {format_expr(sqrt_right)}")

    center = Fraction(-b, 2 * a)
    steps.append(f"{format_expr(x)} = {format_number(center)} +/- {format_expr(sqrt_right)}")
    return steps


def cubic_steps(a: Fraction, b: Fraction, c: Fraction, d: Fraction, var: str) -> List[str]:
    x = Var(var)
    t = Var("t")
    steps: List[str] = []

    standard = build_poly_expr({3: a, 2: b, 1: c, 0: d}, var)
    steps.append(format_equation(standard, Num(Fraction(0))))

    if a != 1:
        steps.append(
            format_equation(
                BinOp(
                    "+",
                    BinOp(
                        "+",
                        BinOp("+", Pow(x, Num(Fraction(3))), BinOp("*", Num(b / a), Pow(x, Num(Fraction(2))))),
                        BinOp("*", Num(c / a), x),
                    ),
                    Num(d / a),
                ),
                Num(Fraction(0)),
            )
        )

    shift = b / (3 * a)
    steps.append(f"{format_expr(x)} = {format_expr(add_constant(t, -shift))}")

    p = (3 * a * c - b * b) / (3 * a * a)
    q = (2 * b * b * b - 9 * a * b * c + 27 * a * a * d) / (27 * a * a * a)
    steps.append(format_equation(BinOp("+", BinOp("+", Pow(t, Num(Fraction(3))), BinOp("*", Num(p), t)), Num(q)), Num(Fraction(0))))
    steps.append(f"p = {format_number(p)}")
    steps.append(f"q = {format_number(q)}")

    delta = (q / 2) * (q / 2) + (p / 3) * (p / 3) * (p / 3)
    steps.append(f"Delta = {format_number(delta)}")

    sqrt_delta = sqrt_fraction(delta)
    u = Pow(BinOp("+", Num(-q / 2), sqrt_delta), Num(Fraction(1, 3)))
    v = Pow(BinOp("-", Num(-q / 2), sqrt_delta), Num(Fraction(1, 3)))
    steps.append(f"u = {format_expr(u)}")
    steps.append(f"v = {format_expr(v)}")
    steps.append(f"{format_expr(t)} = {format_expr(BinOp('+', u, v))}")

    omega = BinOp(
        "+",
        Num(Fraction(-1, 2)),
        BinOp("/", BinOp("*", Imag(), Pow(Num(3), Num(Fraction(1, 2)))), Num(2)),
    )
    omega2 = BinOp(
        "-",
        Num(Fraction(-1, 2)),
        BinOp("/", BinOp("*", Imag(), Pow(Num(3), Num(Fraction(1, 2)))), Num(2)),
    )
    steps.append(f"omega = {format_expr(omega)}")

    shift_expr = Num(shift)
    x1 = BinOp("-", BinOp("+", u, v), shift_expr)
    x2 = BinOp("-", BinOp("+", BinOp("*", u, omega), BinOp("*", v, omega2)), shift_expr)
    x3 = BinOp("-", BinOp("+", BinOp("*", u, omega2), BinOp("*", v, omega)), shift_expr)
    steps.append(f"x1 = {format_expr(x1)}")
    steps.append(f"x2 = {format_expr(x2)}")
    steps.append(f"x3 = {format_expr(x3)}")
    return steps


def linear_form(expr: Expr) -> Optional[Tuple[Fraction, Fraction, Optional[str]]]:
    if isinstance(expr, Num):
        return Fraction(0), expr.value, None
    if isinstance(expr, Imag):
        return None
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
    if isinstance(expr, Pow):
        return None
    if isinstance(expr, BinOp):
        if expr.op in ADD_OPS:
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
        if expr.op in MUL_OPS:
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
    num = value.numerator
    den = value.denominator
    if num < 0:
        return rf"-\frac{{{abs(num)}}}{{{den}}}"
    return rf"\frac{{{num}}}{{{den}}}"


def precedence(expr: Expr) -> int:
    if isinstance(expr, (Num, Var, Imag)):
        return 4
    if isinstance(expr, (Neg, Group)):
        return 4
    if isinstance(expr, Pow):
        return 3
    if isinstance(expr, BinOp):
        if expr.op in MUL_OPS:
            return 2
        return 1
    return 0


def format_expr(expr: Expr) -> str:
    if isinstance(expr, Num):
        return format_number(expr.value)
    if isinstance(expr, Imag):
        return "i"
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
    if isinstance(expr, Pow):
        base = format_expr(expr.base)
        if isinstance(expr.exponent, Num):
            if expr.exponent.value == Fraction(1, 2):
                return f"sqrt({base})"
            if expr.exponent.value == Fraction(1, 3):
                return f"cbrt({base})"
        exponent = format_expr(expr.exponent)
        if precedence(expr.base) < precedence(expr) or isinstance(expr.base, (Neg, BinOp)):
            base = f"({base})"
        if precedence(expr.exponent) < precedence(expr) or isinstance(expr.exponent, (BinOp, Neg)):
            exponent = f"({exponent})"
        return f"{base}^{exponent}"
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
            if isinstance(expr.left, Imag):
                if isinstance(expr.right, Neg):
                    inner = format_expr(expr.right.expr)
                    if precedence(expr.right.expr) < precedence(expr):
                        inner = f"({inner})"
                    return f"-{inner}i"
                return f"{right}i"
            if isinstance(expr.left, Num):
                if expr.left.value == 1:
                    return right
                if expr.left.value == -1:
                    return f"-{right}"
            if isinstance(expr.left, Num) and isinstance(expr.right, Neg):
                inner = format_expr(expr.right.expr)
                if precedence(expr.right.expr) < precedence(expr):
                    inner = f"({inner})"
                return f"-{left}{inner}"
            if isinstance(expr.right, Neg):
                right = f"({right})"
            if isinstance(expr.left, Num) and isinstance(expr.right, Num):
                return rf"{left} \times {right}"
            return f"{left}{right}"
        if expr.op == "/":
            if isinstance(expr.right, Num) and expr.right.value == 1:
                return left
            return rf"\frac{{{left}}}{{{right}}}"
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
    if b != 0:
        if b > 0:
            right_expr: Expr = BinOp("-", Num(b2), Num(b1))
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
    return [(cross_left, cross_right)]


def append_unique(lines: List[str], value: str) -> None:
    if not lines or lines[-1] != value:
        lines.append(value)


def _simplify_expr(expr: Expr, max_steps: int = 20) -> Expr:
    current = expr
    for _ in range(max_steps):
        new = simplify_once(current)
        if new is None or new == current:
            break
        current = new
    return current


def _normalize_equation_str(expr_str: str) -> Optional[str]:
    tokens = tokenize(expr_str)
    split = split_equation(tokens)
    if split is None:
        return None
    left_tokens, right_tokens = split
    left_expr = parse_ratio(left_tokens)
    right_expr = parse_ratio(right_tokens)
    if isinstance(left_expr, Ratio):
        left_expr = BinOp("/", left_expr.left, left_expr.right)
    if isinstance(right_expr, Ratio):
        right_expr = BinOp("/", right_expr.left, right_expr.right)

    left_expr = _simplify_expr(left_expr)
    right_expr = _simplify_expr(right_expr)
    merged = BinOp("+", left_expr, Neg(right_expr))
    merged = _simplify_expr(merged)
    return format_equation(merged, Num(Fraction(0)))


def _split_input_parts(input_str: str) -> List[str]:
    parts: List[str] = []
    for line in input_str.splitlines():
        for chunk in line.split(";"):
            chunk = chunk.strip()
            if chunk:
                parts.append(chunk)
    return parts


def _extract_function_definitions(input_str: str) -> Tuple[List[Tuple[str, str, str]], str]:
    defs: List[Tuple[str, str, str]] = []
    parts = _split_input_parts(input_str)
    remaining: List[str] = []
    for part in parts:
        match = FUNC_DEF_RE.match(part)
        if match:
            name, arg, body = match.group(1), match.group(2), match.group(3).strip()
            defs.append((name, arg, body))
        else:
            remaining.append(part)
    main_expr = remaining[-1] if remaining else ""
    return defs, main_expr


def _pick_placeholder(used: set[str]) -> str:
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        if letter not in used:
            return letter
    return "Z"


def _replace_func_calls(expr: str, name: str, arg: str, replacement: str) -> str:
    pattern = re.compile(rf"{re.escape(name)}\s*\(\s*{re.escape(arg)}\s*\)")
    return pattern.sub(replacement, expr)


def _replace_placeholder_steps(steps: List[str], placeholder: str, name: str, arg: str) -> List[str]:
    pattern = re.compile(rf"(?<![A-Za-z]){re.escape(placeholder)}(?![A-Za-z])")
    return [pattern.sub(f"{name}({arg})", step) for step in steps]


def _solve_raw(input_str: str) -> List[str]:
    tokens = tokenize(input_str)
    split = split_equation(tokens)
    if split is None:
        expr = parse_ratio(tokens)
        if isinstance(expr, Ratio):
            expr = BinOp("/", expr.left, expr.right)
        expr_steps = solve_expression_steps(expr)
        lines: List[str] = []
        for step in expr_steps[1:]:
            append_unique(lines, format_expr(step))
        return lines

    left_tokens, right_tokens = split
    left_expr = parse_ratio(left_tokens)
    right_expr = parse_ratio(right_tokens)

    steps: List[str] = []
    if isinstance(left_expr, Ratio) and isinstance(right_expr, Ratio):
        ratio_steps = solve_ratio_equation_steps(left_expr, right_expr)
        for l, r in ratio_steps:
            append_unique(steps, format_equation(l, r))
        left_expr, right_expr = ratio_steps[-1]
        simplified_steps = simplify_equation_sides(left_expr, right_expr)
        for l, r in simplified_steps:
            append_unique(steps, format_equation(l, r))
        if simplified_steps:
            left_expr, right_expr = simplified_steps[-1]

    diff_expr = BinOp("-", left_expr, right_expr)
    vars_found = find_vars(diff_expr)
    if len(vars_found) == 1:
        var = next(iter(vars_found))
        poly = poly_from_expr(diff_expr, var)
        if poly is not None:
            degree = poly_degree(poly)
            if degree == 2:
                a = poly.get(2, Fraction(0))
                b = poly.get(1, Fraction(0))
                c = poly.get(0, Fraction(0))
                if a != 0:
                    return steps + quadratic_steps(a, b, c, var)
            if degree == 3:
                a = poly.get(3, Fraction(0))
                b = poly.get(2, Fraction(0))
                c = poly.get(1, Fraction(0))
                d = poly.get(0, Fraction(0))
                if a != 0:
                    return steps + cubic_steps(a, b, c, d, var)

    linear_steps = solve_linear_equation_steps(left_expr, right_expr)
    if linear_steps is not None:
        if linear_steps:
            last_left, last_right = linear_steps[-1]
            if is_numeric(last_left) and isinstance(last_right, Var):
                linear_steps.append((last_right, last_left))
        for l, r in linear_steps:
            append_unique(steps, format_equation(l, r))
        return steps

    left_steps = solve_expression_steps(left_expr)
    right_steps = solve_expression_steps(right_expr)
    for l, r in zip(left_steps[1:], right_steps[1:]):
        append_unique(steps, format_equation(l, r))
    return steps


def solve(input_str: str) -> List[str]:
    defs, main_expr = _extract_function_definitions(input_str)
    if not defs:
        return _solve_raw(main_expr)

    used = set(re.findall(r"[A-Za-z]", main_expr))
    placeholders: List[Tuple[str, str, str, str]] = []
    phase1_expr = main_expr
    for name, arg, body in defs:
        placeholder = _pick_placeholder(used)
        used.add(placeholder)
        placeholders.append((name, arg, body, placeholder))
        phase1_expr = _replace_func_calls(phase1_expr, name, arg, placeholder)

    phase1_steps = _solve_raw(phase1_expr)
    for name, arg, _body, placeholder in placeholders:
        phase1_steps = _replace_placeholder_steps(phase1_steps, placeholder, name, arg)

    normalized = _normalize_equation_str(phase1_expr)
    if normalized:
        for name, arg, _body, placeholder in placeholders:
            normalized = _replace_placeholder_steps([normalized], placeholder, name, arg)[0]
        append_unique(phase1_steps, normalized)

    anchor_expr = phase1_steps[-1] if phase1_steps else main_expr
    phase2_expr = anchor_expr
    for name, arg, body, _placeholder in placeholders:
        phase2_expr = _replace_func_calls(phase2_expr, name, arg, f"({body})")
    if phase2_expr == anchor_expr:
        phase2_expr = main_expr
        for name, arg, body, _placeholder in placeholders:
            phase2_expr = _replace_func_calls(phase2_expr, name, arg, f"({body})")

    phase2_steps = _solve_raw(phase2_expr)

    combined: List[str] = []
    for step in phase1_steps:
        append_unique(combined, step)
    for name, arg, body, _placeholder in placeholders:
        append_unique(combined, f"{name}({arg}) = {body}")
    if phase2_expr:
        append_unique(combined, phase2_expr)
    for step in phase2_steps:
        append_unique(combined, step)
    return combined


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
