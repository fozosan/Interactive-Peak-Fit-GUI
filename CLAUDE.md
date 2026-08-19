# CLAUDE Memory — Interactive Peak Fit GUI

## What this is

Cross-platform Tkinter + Matplotlib desktop app for interactive spectral peak
fitting with pseudo-Voigt (Gaussian–Lorentzian) peak models. ALS/polynomial
baseline correction; Classic, Modern TRF and Variable-Projection solvers
(optional LMFIT-VP); asymptotic / bootstrap / Bayesian-MCMC uncertainty; a
single-step "Step ▶" debugger; batch processing; persisted settings. v3.2-beta,
~29.6k lines of Python across `core/` (19), `fit/` (11), `infra/` (4), `batch/`,
`tools/`, `scripts/`, and 166 test files.

Designed by Farhan Zahin. Developed on a Windows machine, so sessions before
2026-08-18 have no transcript here — the written documentation below is the
record of that period.

Tag: **`peak_fit_gui`** (the directory name has spaces, so the tag differs from
it). Confirm with `python3 ~/.local/share/code-wiki/project_tag.py "$PWD"` — never the basename.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).

<!-- CODE-WIKI:START — added by the Code Wiki setup. Delete from START to END to remove cleanly. -->

> **Which machine am I on?** Run
> `python3 ~/.local/share/code-wiki/project_tag.py "$PWD"`.
> **It prints a tag** → the Linux workstation. Everything below applies; carry on.
> **"No such file"** → Windows. This repo reached you over OneDrive sync but the
> Code Wiki did not: the vault lives in `~/Obsidian`, outside OneDrive, and the
> scripts are Linux-only. Skip every graph and vault step below and work from this
> repo's own documentation, source and git history, which all did sync. Do not
> build a vault here — one exists on the Linux box, and a second would diverge.

## Where to look first

Answer in this order and stop as soon as the question is answered. Reading source
to answer a structural question is the failure this ordering exists to prevent.

1. **Graph** — structure. What calls what, what breaks if this changes.
   ```bash
   G="$HOME/Obsidian/Code Wiki/graphify/peak_fit_gui/graphify-out"
   graphify query "<question>" --graph "$G/graph.json"
   ```
   Also `explain "<Node>"`, `path "A" "B"`, `affected "X"`, `god-nodes`.
   `$G/wiki/index.md` is one article per community — 82 of them.
   The hubs are `PeakFitApp` (202 edges), `Peak` (156), `bootstrap_ci()` (68),
   `pv_sum()` (65), `run_batch()` (59). `PeakFitApp` is a 7,113-line single file;
   expect most GUI questions to land there and use `explain` rather than reading it.
2. **Written documentation** — the table below. Two of these are large and
   genuinely comprehensive; prefer them over re-deriving from source.
3. **Vault** — `~/Obsidian/Code Wiki/permanent/` for *why* a decision was made,
   the vault-root `logs/` for what the last session left unfinished. Past sessions
   are in `chats/code/` tagged `project:` — grep for specifics, never bulk-read.
   Nothing there predates 2026-08-18; that work was on the other machine.
4. **Raw code** — only when editing, or when the layers above disagree.

## Documentation

In the repo and staying there — these predate the Code Wiki and are referenced by
each other and by the README.

| What | Where |
|---|---|
| Complete codebase documentation — architecture, modules, data flow | `CODEBASE_KNOWLEDGE.md` (36 KB) |
| Comprehensive analysis with per-module line counts and statistics | `COMPREHENSIVE_CODEBASE_SUMMARY.md` (48 KB) |
| User-facing overview, screenshot, install and usage | `README.md` |
| The 6 refactoring tasks, duplication reduction, patterns established | `REFACTORING_SUMMARY.md` |
| Theme system implementation | `THEME_IMPLEMENTATION_SUMMARY.md` |
| Font rendering — design and knowledge base | `font_rendering_plan.md`, `font_rendering_kb.md` |
| Apply-path performance notes | `PERFORMANCE_APPLY_KB.md` |
| Release history | `docs/CHANGELOG.md` |

## Adding new documentation

New long-form documentation goes to `~/Obsidian/Code Wiki/projects/peak_fit_gui/`,
not into this repo — one living document per subject, revised in place. The files
above are the exception and stay where they are.

Do not write project docs inline — ask, and it is delegated to the
**`wiki-doc-writer` agent** (sonnet). It queries the graph before reading source,
writes to the vault, and returns only the paths. Session logs and `permanent/`
notes are the exception: their input exists only in the live conversation, so
`/wiki-save` writes those in the working model.

Naming for **new** files:

| What | Where |
|---|---|
| Figures and run outputs | `data/<context>_<YYYYMMDD>_<HHMMSS>/` |
| A prototype's report | `features/<feature>_prototype/` |
| A normal script's docs | `features/<context>_script/` |

**Existing files keep their original names — never rename them to fit this
convention.** They cross-reference each other; renaming silently breaks that.

Render the vault docs to a versioned `.docx` in this repo's `docs/`:

```bash
python3 ~/.local/share/code-wiki/wiki_docx.py peak_fit_gui
```

Verify the render — pandoc drops unresolvable images with only a warning and
still exits 0. The rendered count must be at least the number of `![](...)` links:

```bash
python3 -c "import zipfile,sys;print(zipfile.ZipFile(sys.argv[1]).read('word/document.xml').decode('utf8','replace').count('<w:drawing>'))" docs/peak_fit_gui_docs_v1.docx
```

Full rules, including the figure and docx conventions: `~/.claude/CLAUDE.md`.

Session start: `/wiki-resume`. Session end: `/wiki-save`.
Documentation requests go to the `wiki-doc-writer` agent, never inline.

<!-- CODE-WIKI:END -->
