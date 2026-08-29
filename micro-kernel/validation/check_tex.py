"""LaTeX pre-flight for the three manuscripts.

    python check_tex.py

Checks, for each paper: that every \\cite key is defined in its
references.bib, that every \\Cref/\\ref target has a matching \\label,
that theorem-like environments are balanced, and that no \\label is
duplicated. Exits non-zero if anything is wrong.
"""

from __future__ import annotations

import pathlib
import re
import sys

DOCS = pathlib.Path(__file__).resolve().parent.parent / "docs"

PAPERS = [
    "heihachi-runtime-graph",
    "mishima-propagator",
    "sangoma-instrument-synthesizer",
]

CITE = re.compile(r"\\cite[tp]?\*?\{([^}]*)\}")
LABEL = re.compile(r"\\label\{([^}]*)\}")
LSTLABEL = re.compile(r"label=\{?([A-Za-z0-9:_\-]+)\}?")
REF = re.compile(r"\\(?:Cref|cref|ref|autoref)\{([^}]*)\}")
BIBKEY = re.compile(r"@\w+\{([^,]+),")
BEGIN = re.compile(r"\\begin\{(\w+)\}")
END = re.compile(r"\\end\{(\w+)\}")


def check(paper: str) -> list[str]:
    base = DOCS / paper
    tex_path = base / f"{paper}.tex"
    bib_path = base / "references.bib"
    problems: list[str] = []

    if not tex_path.exists():
        return [f"{paper}: missing {tex_path.name}"]
    tex = tex_path.read_text(encoding="utf-8")
    bib = bib_path.read_text(encoding="utf-8") if bib_path.exists() else ""

    cited: set[str] = set()
    for m in CITE.finditer(tex):
        for key in m.group(1).split(","):
            if key.strip():
                cited.add(key.strip())
    defined = {k.strip() for k in BIBKEY.findall(bib)}

    for key in sorted(cited - defined):
        problems.append(f"{paper}: cited but not in references.bib: {key}")

    # listings carry their label inside the optional argument
    labels = LABEL.findall(tex) + LSTLABEL.findall(tex)
    label_set = set(labels)
    for lab in sorted({x for x in labels if labels.count(x) > 1}):
        problems.append(f"{paper}: duplicate label: {lab}")

    referenced: set[str] = set()
    for m in REF.finditer(tex):
        for key in m.group(1).split(","):
            if key.strip():
                referenced.add(key.strip())
    for key in sorted(referenced - label_set):
        problems.append(f"{paper}: reference to missing label: {key}")

    opened = BEGIN.findall(tex)
    closed = END.findall(tex)
    for env in sorted(set(opened) | set(closed)):
        if opened.count(env) != closed.count(env):
            problems.append(
                f"{paper}: unbalanced environment {env!r}: "
                f"{opened.count(env)} begin, {closed.count(env)} end")

    stats = (f"{paper}: {len(tex.splitlines())} lines, {len(cited)} keys cited, "
             f"{len(defined)} defined, {len(label_set)} labels, "
             f"{len(cited - defined)} missing")
    print(("FAIL  " if problems else "ok    ") + stats)
    if defined - cited:
        print("      unused bib keys: " + ", ".join(sorted(defined - cited)))
    return problems


def main() -> int:
    all_problems: list[str] = []
    for paper in PAPERS:
        all_problems.extend(check(paper))
    if all_problems:
        print()
        for p in all_problems:
            print("  " + p)
        return 1
    print("\nall papers pass pre-flight")
    return 0


if __name__ == "__main__":
    sys.exit(main())
