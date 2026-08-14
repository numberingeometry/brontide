#!/usr/bin/env python3
"""verify_paper_compiles.py -- a compile check that can actually fail.

Run:  python scripts/verify_paper_compiles.py [path/to/ai4metascience.tex]

WHY THIS EXISTS. The obvious check --

    pdflatex -interaction=nonstopmode X.tex ; exit 0 if X.pdf exists and rc == 0

-- is close to vacuous, for two independent reasons:

  1. `-interaction=nonstopmode` tells pdflatex not to stop at an error prompt. It
     still reports many error classes while returning 0.
  2. `os.path.exists(X.pdf)` is satisfied by a PDF left behind by ANY earlier
     build, including one from before the errors were introduced.

So that check passes on a document that does not compile. This one builds in a
scratch copy from a deleted-artifact state, runs the full bibtex cycle, and reads
the FRESH log for the things that actually matter:

  * lines beginning with "!"                 -> LaTeX errors
  * "LaTeX Warning: Reference ... undefined" -> broken \\ref
  * "Citation ... undefined"                 -> broken \\cite
  * "There were undefined references"
  * whether the cross-reference fixpoint is reached (no "Label(s) may have
    changed" on the final pass) -- if it is not, the numbers a reader sees are
    not the numbers the .aux holds
  * overfull boxes beyond a stated tolerance
  * that the PDF is newly written by THIS run, not inherited

Exit 0 only if there are no errors, no undefined references or citations, and the
cross-reference fixpoint is reached. Overfull boxes are reported, not fatal.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

OVERFULL_TOL_PT = 20.0
_RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    _RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    return bool(ok)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python scripts/verify_paper_compiles.py PAPER.tex", file=sys.stderr)
        return 2
    tex = Path(sys.argv[1])
    print("verify_paper_compiles.py -- clean-room build, and a check that can fail")
    print(f"source = {tex}")
    if not tex.exists():
        print("  source .tex not found")
        return 1

    src = tex.parent
    stem = tex.stem
    with tempfile.TemporaryDirectory(prefix="papercheck-") as tmp:
        work = Path(tmp)
        # copy only what a build needs; never touch the original directory
        for pat in ("*.tex", "*.sty", "*.bib", "*.bst", "*.pdf", "*.png", "*.jpg", "*.cls"):
            for f in src.glob(pat):
                if f.is_file():
                    shutil.copy2(f, work / f.name)
        # delete every build product so nothing can be inherited
        for pat in ("*.aux", "*.bbl", "*.blg", "*.out", "*.log", "*.toc", "*.brf"):
            for f in work.glob(pat):
                f.unlink()
        (work / f"{stem}.pdf").unlink(missing_ok=True)
        print(f"  clean room = {work}")
        check("no inherited PDF at the start of the build",
              not (work / f"{stem}.pdf").exists())

        seq = [("pdflatex", 1), ("bibtex", 1), ("pdflatex", 1), ("pdflatex", 1), ("pdflatex", 1)]
        rcs = []
        for prog, _ in seq:
            args = ([prog, "-interaction=nonstopmode", f"{stem}.tex"] if prog == "pdflatex"
                    else [prog, stem])
            r = subprocess.run(args, cwd=work, capture_output=True, text=True,
                               errors="replace", timeout=600)
            rcs.append((prog, r.returncode))
        print(f"  build sequence returncodes: {rcs}")

        pdf = work / f"{stem}.pdf"
        check("a PDF was produced by THIS build", pdf.exists(),
              f"{pdf.stat().st_size} bytes" if pdf.exists() else "absent")

        log = (work / f"{stem}.log")
        if not log.exists():
            check("a fresh log was produced", False)
            return 1
        txt = log.read_text(encoding="utf-8", errors="replace")

        errs = [l for l in txt.splitlines() if l.startswith("!")]
        undef_ref = re.findall(r"LaTeX Warning: Reference [^\n]*undefined", txt)
        undef_cite = re.findall(r"(?:LaTeX Warning: )?Citation [^\n]*undefined", txt)
        undef_any = "There were undefined references" in txt
        rerun = "Label(s) may have changed" in txt
        overfull = [(float(m.group(1)), m.group(0))
                    for m in re.finditer(r"Overfull \\hbox \((\d+\.\d+)pt too wide\)[^\n]*", txt)]
        big = [o for o in overfull if o[0] > OVERFULL_TOL_PT]
        pages = re.findall(r"Output written on [^(]*\((\d+) pages", txt)

        for e in errs[:10]:
            print(f"    ERROR  {e}")
        for u in (undef_ref + undef_cite)[:10]:
            print(f"    UNDEF  {u}")

        ok_err = check(f"0 LaTeX errors in the fresh log", not errs, f"{len(errs)} found")
        ok_ref = check("0 undefined references", not undef_ref, f"{len(undef_ref)} found")
        ok_cit = check("0 undefined citations", not undef_cite and not undef_any,
                       f"{len(undef_cite)} found; 'There were undefined references' = {undef_any}")
        ok_fix = check("cross-reference fixpoint reached on the final pass (no rerun warning)",
                       not rerun,
                       "log still says 'Label(s) may have changed' -- displayed numbers may be stale"
                       if rerun else "")
        print(f"  pages: {pages[-1] if pages else '?'}")
        print(f"  overfull hboxes > {OVERFULL_TOL_PT}pt: {len(big)}"
              + (f"   worst {max(o[0] for o in big):.1f}pt" if big else ""))
        for _, line in sorted(big, reverse=True)[:5]:
            print(f"    {line}")
        check("overfull boxes are reported (not fatal)", True,
              f"{len(big)} beyond {OVERFULL_TOL_PT}pt")

        # A self-test, so that a PASS above means something.
        #
        # CORRECTED after running the first version: pdflatex returns 1 on a hard error
        # (an undefined control sequence), so the naive "rc == 0 and pdf exists" check
        # DOES catch those. Claiming it was vacuous in general was wrong.
        #
        # The gap that actually matters is undefined \ref and \cite, and a cross-reference
        # fixpoint that never converges. pdflatex exits 0 on all of those while the
        # document is broken for a reader, so the self-test uses a document with only
        # those defects and nothing else.
        bad = work / "selftest_broken.tex"
        bad.write_text("\\documentclass{article}\n\\begin{document}\n"
                       "See \\ref{nosuchlabel} and \\cite{nosuchcite}.\n"
                       "\\end{document}\n", encoding="utf-8")
        rb = subprocess.run(["pdflatex", "-interaction=nonstopmode", "selftest_broken.tex"],
                            cwd=work, capture_output=True, text=True, errors="replace", timeout=300)
        btxt = (work / "selftest_broken.log").read_text(encoding="utf-8", errors="replace")
        b_pdf = (work / "selftest_broken.pdf").exists()
        b_ref = re.findall(r"LaTeX Warning: Reference [^\n]*undefined", btxt)
        b_cit = re.findall(r"Citation [^\n]*undefined", btxt)
        print("\n  SELF-TEST on a document whose only defects are an undefined ref and cite:")
        print(f"    pdflatex returncode = {rb.returncode}, pdf produced = {b_pdf}, "
              f"undefined refs = {len(b_ref)}, undefined cites = {len(b_cit)}")
        check("SELF-TEST: the naive 'rc==0 and pdf exists' check PASSES a document with "
              "undefined references and citations, so it cannot back the claim; reading the "
              "log catches what it misses",
              rb.returncode == 0 and b_pdf and (len(b_ref) + len(b_cit)) > 0,
              f"naive check returns 0 here, while the log shows {len(b_ref)} undefined "
              f"reference(s) and {len(b_cit)} undefined citation(s)")

    print("\n" + "=" * 78)
    n_fail = sum(1 for _, ok, _ in _RESULTS if not ok)
    for name, ok, detail in _RESULTS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    hard = not (ok_err and ok_ref and ok_cit and ok_fix)
    print(f"\n  {len(_RESULTS) - n_fail}/{len(_RESULTS)} checks passed.")
    print("  OVERALL:", "FAIL" if hard else "PASS")
    return 1 if hard else 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    raise SystemExit(main())
