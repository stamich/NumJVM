"""Summarises JaCoCo line and branch coverage per module.

Reads build/reports/jacoco/test/jacocoTestReport.csv of each module given on the
command line and prints a Markdown table. On GitHub Actions the table goes to the job
summary instead, and stdout gets one notice annotation per module, so the numbers also
appear on the workflow run and in the pull request checks.

Usage:
  python3 scripts/coverage_summary.py numjvm-core numjvm-io
"""

import csv
import os
import sys


def totals(path):
    lines = [0, 0]
    branches = [0, 0]
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lines[0] += int(row["LINE_COVERED"])
            lines[1] += int(row["LINE_MISSED"])
            branches[0] += int(row["BRANCH_COVERED"])
            branches[1] += int(row["BRANCH_MISSED"])
    return lines, branches


def percent(covered, missed):
    total = covered + missed
    return 100.0 * covered / total if total else 100.0


def main():
    modules = sys.argv[1:]
    if not modules:
        print(__doc__)
        return 2
    table = ["| Module | Line coverage | Branch coverage |", "|---|---:|---:|"]
    notices = []
    for module in modules:
        path = os.path.join(module, "build", "reports", "jacoco", "test", "jacocoTestReport.csv")
        if not os.path.exists(path):
            table.append(f"| {module} | no report | |")
            continue
        (lc, lm), (bc, bm) = totals(path)
        line = f"{percent(lc, lm):.1f} % ({lc}/{lc + lm})"
        branch = f"{percent(bc, bm):.1f} % ({bc}/{bc + bm})"
        table.append(f"| {module} | {line} | {branch} |")
        notices.append(f"::notice title=Coverage {module}::lines {line}, branches {branch}")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write("\n".join(table) + "\n")
        print("\n".join(notices))
    else:
        print("\n".join(table))
    return 0


if __name__ == "__main__":
    sys.exit(main())
