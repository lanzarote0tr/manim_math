from solver import ParseError, solve


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
