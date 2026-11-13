"""Build the marimo versions of the course notebooks in marimo/.

The .ipynb notebooks are the source; never edit marimo/*.py by hand, re-run this instead:

    uv run python course/make_marimo.py            # all notebooks
    uv run python course/make_marimo.py t02_neural_networks_enzymes.ipynb

On top of `marimo convert` it
- hides the code of cells whose first line is `# @title ...` (the support code that is
  collapsed in the notebooks), and
- replaces the ipywidgets slider cells of the playgrounds with marimo slider forms.
"""
import ast
import re
import textwrap
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "marimo"

SLIDER_MD_OLD = ("*Sliders:* in Jupyter or Colab with `ipywidgets` installed, the next cell shows sliders and "
                 "redraws the figure when you release one. Without `ipywidgets`, just call the function with other values.")
SLIDER_MD_NEW = ("*Sliders:* set the sliders below and press **Submit** to redraw the figure. "
                 "You can also call the function directly with other values.")


def logsteps(lo, hi):
    """1, 2, 5 x 10^k from 10^lo to 10^hi (marimo sliders have no log scale)."""
    return [float(f"{m * 10.0 ** k:.0e}") for k in range(lo, hi + 1) for m in (1, 2, 5)
            if m * 10.0 ** k <= 10.0 ** hi * 1.0001]


# notebook -> (playground function, name of the form, [(argument, marimo widget)])
PLAYGROUNDS = {
    "t01_toxicity_qsar": ("poly_playground", "poly_controls", [
        ("degree", "mo.ui.slider(0, 20, value=3, label='degree')"),
        ("n_train", "mo.ui.slider(8, 300, step=4, value=20, label='n_train')"),
        ("noise", "mo.ui.slider(0.0, 0.8, step=0.05, value=0.2, label='noise')"),
        ("alpha", f"mo.ui.slider(steps={logsteps(-8, 2)}, value=1e-08, label='alpha')"),
        ("seed", "mo.ui.slider(0, 20, value=0, label='seed')"),
        ("show_variance", "mo.ui.checkbox(value=True, label='show_variance')"),
    ]),
    "t02_neural_networks_enzymes": ("mlp_playground", "mlp_controls", [
        ("width", "mo.ui.slider(2, 500, step=2, value=200, label='width')"),
        ("layers", "mo.ui.slider(1, 4, value=2, label='layers')"),
        ("alpha", f"mo.ui.slider(steps={logsteps(-6, 1)}, value=1e-05, label='alpha')"),
        ("n_train", "mo.ui.slider(10, 300, step=5, value=20, label='n_train')"),
        ("noise", "mo.ui.slider(0.0, 0.6, step=0.05, value=0.2, label='noise')"),
        ("epochs", "mo.ui.slider(100, 6000, step=100, value=3000, label='epochs')"),
        ("seed", "mo.ui.slider(0, 20, value=0, label='seed')"),
    ]),
    "t03_rnn_protein_targeting": ("gru_playground", "gru_controls", [
        ("hidden", "mo.ui.slider(2, 256, step=2, value=32, label='hidden')"),
        ("n_train", "mo.ui.slider(30, 2700, step=30, value=300, label='n_train')"),
        ("epochs", "mo.ui.slider(5, 100, step=5, value=60, label='epochs')"),
        ("lr", f"mo.ui.slider(steps={logsteps(-4, -1)}, value=0.01, label='lr')"),
        ("weight_decay", f"mo.ui.slider(steps={logsteps(-6, -1)}, value=1e-06, label='weight_decay')"),
        ("seed", "mo.ui.slider(0, 20, value=0, label='seed')"),
    ]),
}


def replace_sliders(src, stem):
    func, ctl, sliders = PLAYGROUNDS[stem]
    assert src.count(SLIDER_MD_OLD) == 1, stem
    src = src.replace(SLIDER_MD_OLD, SLIDER_MD_NEW)
    pat = re.compile(r"@app\.cell(?:\(hide_code=True\))?\ndef _\(" + func + r"\):\n(    # @title [^\n]*\n)?"
                     r"    try:\n        from ipywidgets.*?\n    return\n", re.S)
    matches = list(pat.finditer(src))
    assert len(matches) == 1, stem
    m = matches[0]
    title = m.group(1) or ""
    items = "".join(f"        {name!r}: {widget},\n" for name, widget in sliders)
    new = (
        "@app.cell\n"
        "def _(mo):\n"
        f"{title}"
        f"    {ctl} = mo.ui.dictionary({{\n{items}    }}).form(submit_button_label='Submit')\n"
        f"    {ctl}\n"
        f"    return ({ctl},)\n"
        "\n\n"
        "@app.cell\n"
        f"def _({ctl}, {func}):\n"
        f"{title}"
        f"    if {ctl}.value is not None:\n"
        f"        {func}(**{ctl}.value)\n"
        "    return\n"
    )
    return src[:m.start()] + new + src[m.end():]


def hide_titled_cells(src):
    """`@app.cell` -> `@app.cell(hide_code=True)` when the body starts with `# @title`.

    `marimo convert` already does this for cells with jupyter.source_hidden; this also covers
    the slider cells written above and any titled cell that lacks the metadata."""
    return re.sub(r"@app\.cell\n(def _\([^)]*\):\n    # @title )", r"@app.cell(hide_code=True)\n\1", src)


def replace_display(src):
    """Jupyter's built-in display() does not exist in marimo: use mo.output.append()."""
    def fix(m):
        args, body = m.group(1), m.group(2)
        if not re.search(r"^\s+display\(", body, re.M):
            return m.group(0)
        body = re.sub(r"^(\s+)display\(", r"\1mo.output.append(", body, flags=re.M)
        names = [a.strip() for a in args.split(",") if a.strip() and a.strip() != "display"]
        if "mo" not in names:
            names = sorted(names + ["mo"])
        return f"def _({', '.join(names)}):\n{body}"
    return re.sub(r"def _\(([^)]*)\):\n(.*?\n    return[^\n]*\n)", fix, src, flags=re.S)


PLOT_NAMES = re.compile(r"^_?(plt|fig|figure|ax\d*|axes|cs)$")
PLOT_METHODS = re.compile(r"^(plot|hist|bar|barh|scatter|imshow|legend|colorbar|text|annotate|title|"
                          r"tight_layout|axhline|axvline|axvspan|fill_between|set_\w+|x\w*|y\w*|grid)$")
DRAWS = re.compile(r"\bplt\.|\bplot\w*\(|\.(plot|hist|bar)\b|\bax\d*\b|\baxes\b")


def _is_plot_call(node):
    """Is this expression a matplotlib call (plt.legend(), ax.set_xlabel(...), df.plot(...))?"""
    if not isinstance(node, ast.Call):
        return False
    f = node.func
    while isinstance(f, ast.Attribute):
        if PLOT_METHODS.match(f.attr):
            return True
        f = f.value
        if isinstance(f, ast.Call):
            f = f.func
    # plt.*, ax.*, fig.*, or a plotting helper of the notebook (plot_confusion, plot_reading, ...)
    return isinstance(f, ast.Name) and bool(PLOT_NAMES.match(f.id) or "plot" in f.id)


def show_figures(src):
    """marimo shows only a cell's last expression, not every figure the cell drew as Jupyter does.

    Add plt.show() to each cell that draws: after the last statement, or just before it when the
    cell ends with a value to display (a table), so that both appear."""
    def fix(m):
        head, args, body = m.group(1), m.group(2), m.group(3)
        lines = body.split("\n")
        code = textwrap.dedent("\n".join(lines[:-2]))           # drop "    return ..." and ""
        if not DRAWS.search(code) or "mo.ui." in code or code.lstrip().startswith("mo.md("):
            return m.group(0)
        stmts = ast.parse(code).body
        if not stmts or (isinstance(stmts[-1], ast.Expr) and "plt.show()" in ast.unparse(stmts[-1])):
            return m.group(0)
        last = stmts[-1]
        if isinstance(last, (ast.FunctionDef, ast.ClassDef)):
            return m.group(0)                                    # only defines a helper
        before = isinstance(last, ast.Expr) and not _is_plot_call(last.value)
        at = last.lineno - 1 if before else last.end_lineno
        lines.insert(at, "    plt.show()")
        names = [a.strip() for a in args.split(",") if a.strip()]
        if "plt" in names:
            signature = f"def _({args}):\n"
        else:
            signature = f"def _({', '.join(sorted(names + ['plt']))}):\n"
        return head + signature + "\n".join(lines)
    return re.sub(r"(@app\.cell[^\n]*\n)def _\(([^)]*)\):\n(.*?\n    return[^\n]*\n)", fix, src, flags=re.S)


def build(nb):
    OUT.mkdir(exist_ok=True)
    out = OUT / (nb.stem + ".py")
    subprocess.run(["marimo", "convert", str(nb), "-o", str(out)], check=True, capture_output=True)
    src = out.read_text()
    if nb.stem in PLAYGROUNDS:
        src = replace_sliders(src, nb.stem)
    src = hide_titled_cells(show_figures(replace_display(src)))
    out.write_text(src)
    print(f"{nb.name} -> {out.relative_to(HERE)} ({src.count('# @title')} support cells with hidden code)")


if __name__ == "__main__":
    names = sys.argv[1:] or sorted(p.name for p in HERE.glob("*.ipynb"))
    for name in names:
        build(HERE / Path(name).name)
