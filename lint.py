#!/usr/bin/env python3
"""
Linting script for wakegen project.

This script runs all linting and formatting tools in sequence.
Use --fix to automatically fix issues where possible.
"""
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


def run_command(cmd: List[str], description: str) -> Tuple[bool, str]:
    """Run a command and return success status and output."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=Path(__file__).parent
        )

        # Print output
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)

        success = result.returncode == 0
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"\n{status}: {description}")

        return success, result.stdout + result.stderr
    except Exception as e:
        print(f"❌ ERROR: {e}", file=sys.stderr)
        return False, str(e)


def main() -> int:
    """Run all linting checks."""
    fix_mode = "--fix" in sys.argv

    print("🔍 Wakegen Linting Suite")
    print(f"Mode: {'FIX' if fix_mode else 'CHECK'}")

    results = []

    # 1. Black - Code formatting
    if fix_mode:
        success, _ = run_command(["black", "."], "Black - Auto-format code")
    else:
        success, _ = run_command(
            ["black", "--check", "."], "Black - Check code formatting"
        )
    results.append(("Black", success))

    # 2. isort - Import sorting
    if fix_mode:
        success, _ = run_command(["isort", "."], "isort - Auto-sort imports")
    else:
        success, _ = run_command(
            ["isort", "--check-only", "."], "isort - Check import order"
        )
    results.append(("isort", success))

    # 3. Ruff - Fast modern linter (can auto-fix many issues)
    if fix_mode:
        success, _ = run_command(
            ["ruff", "check", "wakegen", "--fix"], "Ruff - Lint and auto-fix"
        )
    else:
        success, _ = run_command(["ruff", "check", "wakegen"], "Ruff - Fast linting")
    results.append(("Ruff", success))

    # 4. mypy - Type checking (no auto-fix)
    success, _ = run_command(["mypy", "wakegen"], "mypy - Type checking")
    results.append(("mypy", success))

    # 5. Flake8 - PEP 8 style guide (no auto-fix)
    success, _ = run_command(
        ["flake8", "wakegen", "--max-line-length=88", "--extend-ignore=E203,W503"],
        "Flake8 - PEP 8 style checking",
    )
    results.append(("Flake8", success))

    # Print summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)

    all_passed = True
    for tool, success in results:
        status = "✅" if success else "❌"
        print(f"{status} {tool}")
        if not success:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("🎉 All checks passed!")
        return 0
    else:
        print("⚠️  Some checks failed. Run with --fix to auto-fix issues.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
