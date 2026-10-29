"""Build the static web version of the vol lab: one folder, no server, no account.

Run from the repository root:

    python scripts/build_public_vollab.py OUTPUT_DIR [--cdn]

The folder runs ``dashboard/vol_lab.py`` in the visitor's browser with Pyodide
(``marimo export html-wasm --mode run``). Next to the page, in ``public/``, it ships
the pure-Python ``volsurface`` wheel built from this checkout and the SPX fits in
``data/presets/spx_fits.npz`` and every model's fitted parameters in
``data/presets/spx_model_fits.json`` (14 October 2026 SPXW snapshot), plus the plotly wheel
pinned in ``uv.lock`` (Pyodide does not ship plotly). Nothing calls back to
any server of ours.

By default the build also self-hosts the runtime: the Pyodide core files and every
package the page loads (from the jsDelivr Pyodide CDN and PyPI, hash-checked against
marimo's Pyodide lockfile) go in ``pyodide/``, and the MathJax file plotly loads goes in
``vendor/``, so a visitor's browser fetches nothing from any other host. ``--cdn`` skips
this and leaves the page loading them from cdn.jsdelivr.net, wasm.marimo.app and
files.pythonhosted.org (a smaller folder, but third-party requests).

Serve the folder with any static file server, for example
``python -m http.server -d OUTPUT_DIR``; opening ``index.html`` from disk does not work.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOADER = ROOT / "scripts" / "public_vollab"
REPO_URL = "https://github.com/coreyjpritchard/volsurface"
WHEEL = "volsurface-0.1.0-py3-none-any.whl"
# Phone page: hide marimo's own controls and badge, use the full width.
MOBILE_CSS = """
div.fixed.right-0.top-0.z-50, div.fixed.bottom-0.right-0.z-50 { display: none !important; }
html, body { overflow-x: hidden; }
"""
CACHE = Path.home() / ".cache" / "volsurface-web"
# Packages the page imports: marimo's boot set, jedi (imported at kernel start), pyodide-http,
# and ours. marimo also asks for Pygments and docutils (traceback colouring and docstrings
# in the editor); a read-only app without tracebacks does not use them, so the build drops
# that request (1.9 MB).
RUNTIME_PACKAGES = [
    "micropip",
    "msgspec",
    "marimo-base",
    "markdown",
    "pymdown-extensions",
    "narwhals",
    "packaging",
    "jedi",
    "parso",
    "pyodide-http",
    "numpy",
    "scipy",
    "pandas",
    "tzdata",
]
PYODIDE_CORE = ["pyodide.asm.mjs", "pyodide.asm.wasm", "python_stdlib.zip"]


def fetch(url: str, sha256: str | None = None) -> bytes:
    """Download ``url`` once (cached under ~/.cache/volsurface-web), checking sha256 if given."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cached = CACHE / hashlib.sha256(url.encode()).hexdigest()
    if cached.exists():
        data = cached.read_bytes()
    else:
        # Some hosts refuse urllib's default User-Agent.
        req = urllib.request.Request(url, headers={"User-Agent": "volsurface-build"})
        with urllib.request.urlopen(req) as resp:
            data = resp.read()
        cached.write_bytes(data)
    if sha256 and hashlib.sha256(data).hexdigest() != sha256:
        cached.unlink()
        raise SystemExit(f"hash mismatch for {url}")
    return data


def self_host(out: Path) -> None:
    """Copy the Pyodide runtime, its packages and MathJax into ``out``; point the page at them."""
    assets = list((out / "assets").glob("*.js"))
    text = {f: f.read_text() for f in assets}
    (pyodide_version,) = {
        v
        for t in text.values()
        for v in re.findall(r"cdn\.jsdelivr\.net/pyodide/(v[\d.]+)/full/", t)
    }
    marimo_version = subprocess.check_output(
        [sys.executable, "-c", "import marimo; print(marimo.__version__)"], text=True
    ).strip()
    cdn = f"https://cdn.jsdelivr.net/pyodide/{pyodide_version}/full/"
    lock = json.loads(
        fetch(
            f"https://wasm.marimo.app/pyodide-lock.json?v={marimo_version}&pyodide={pyodide_version}"
        )
    )
    packages = lock["packages"]
    need: set[str] = set()
    todo = list(RUNTIME_PACKAGES)
    while todo:
        name = todo.pop()
        if name not in need:
            need.add(name)
            todo.extend(packages[name].get("depends", []))
    dest = out / "pyodide"
    dest.mkdir()
    for name in PYODIDE_CORE:
        (dest / name).write_bytes(fetch(cdn + name))
    kept = {}
    for name in sorted(need):
        entry = dict(packages[name])
        url = entry["file_name"]
        if not url.startswith("https://"):
            url = cdn + url
        filename = url.rsplit("/", 1)[1]
        (dest / filename).write_bytes(fetch(url, entry.get("sha256")))
        entry["file_name"] = filename
        kept[name] = entry
    (dest / "pyodide-lock.json").write_text(json.dumps({"info": lock["info"], "packages": kept}))
    for index, prefix in ((out / "index.html", "./"), (out / "m" / "index.html", "../")):
        html = index.read_text()
        tag = '<marimo-wasm hidden="">'
        if html.count(tag) != 1:
            raise SystemExit(f"could not find the marimo-wasm element in {index}")
        index.write_text(
            html.replace(
                tag,
                f'<marimo-wasm hidden="" data-pyodide-index-url="{prefix}pyodide/" '
                f'data-pyodide-lockfile-url="{prefix}pyodide/pyodide-lock.json">',
            )
        )
    # marimo's plotly view waits for MathJax (2.1 MB) before drawing, to typeset LaTeX in
    # figure text. No figure here uses LaTeX, so it gets an empty stand-in, next to each page
    # because the script path is relative to the page.
    mathjax = {
        u
        for t in text.values()
        for u in re.findall(r"https://cdn\.jsdelivr\.net/npm/mathjax[^`'\"]+", t)
    }
    for url in mathjax:
        local = "vendor/mathjax/" + url.rsplit("/", 1)[1]
        for page_dir in (out, out / "m"):
            (page_dir / local).parent.mkdir(parents=True, exist_ok=True)
            (page_dir / local).write_text("/* MathJax is not loaded: no figure uses LaTeX. */\n")
        for f, t in text.items():
            if url in t:
                text[f] = t = t.replace(url, "./" + local)
                f.write_text(t)
    # Drop the editor-only packages from marimo's dependency request (see RUNTIME_PACKAGES).
    editor_only = "e=`import docutils\\n${e}`,e=`import pygments\\n${e}`,"
    patched = 0
    for f, t in text.items():
        if editor_only in t:
            text[f] = t = t.replace(editor_only, "")
            f.write_text(t)
            patched += 1
    if not patched:
        raise SystemExit("marimo's editor package request not found; check the marimo version")


def slim_plotly(path: Path) -> None:
    """Drop plotly.js copies and the Jupyter extension from the plotly wheel (6 MB).

    marimo draws figures with its own plotly.js; the Python package only builds the figure
    JSON, so the bundled JavaScript is never read in the browser.
    """
    drop = (
        "plotly/package_data/plotly.min.js",
        "plotly/package_data/widgetbundle.js",
        "plotly/labextension/",
        ".data/data/share/jupyter/",
    )
    tmp = path.with_suffix(".tmp")
    with zipfile.ZipFile(path) as src, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as dst:
        for info in src.infolist():
            if any(d in info.filename for d in drop):
                continue
            data = src.read(info)
            if info.filename.endswith(".dist-info/RECORD"):
                data = "".join(
                    line
                    for line in data.decode().splitlines(keepends=True)
                    if not any(d in line for d in drop)
                ).encode()
            dst.writestr(info, data)
    tmp.replace(path)


def locked_wheel(name: str, dest: Path) -> str:
    """Download the pure-Python wheel of ``name`` pinned in uv.lock and check its hash."""
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    (pkg,) = [p for p in lock["package"] if p["name"] == name]
    (wheel,) = [w for w in pkg["wheels"] if w["url"].endswith("-py3-none-any.whl")]
    filename = wheel["url"].rsplit("/", 1)[1]
    with urllib.request.urlopen(wheel["url"]) as resp:
        data = resp.read()
    algo, digest = wheel["hash"].split(":")
    if hashlib.new(algo, data).hexdigest() != digest:
        raise SystemExit(f"hash mismatch for {filename}")
    (dest / filename).write_bytes(data)
    return filename


# Narrow screens go to the phone layout unless the visitor chose the desktop one.
DESKTOP_HEAD = """<script>
(function () {
  var q = new URLSearchParams(location.search);
  try {
    if (q.has("desktop")) localStorage.setItem("vollab-view", "desktop");
    if (localStorage.getItem("vollab-view") === "desktop") return;
  } catch (e) {}
  if (window.matchMedia("(max-width: 760px)").matches) location.replace("m/");
})();
</script>
"""
MOBILE_HEAD = """<script>
try { localStorage.removeItem("vollab-view"); } catch (e) {}
</script>
<style>
MOBILE_CSS
</style>
"""


def export(notebook: Path, out: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "marimo",
            "export",
            "html-wasm",
            str(notebook),
            "-o",
            str(out),
            "--mode",
            "run",
            "--no-show-code",
        ],
        check=True,
    )


def add_head(index: Path, snippet: str) -> None:
    html = index.read_text()
    if html.count("</head>") != 1:
        raise SystemExit(f"no </head> in {index}")
    index.write_text(html.replace("</head>", snippet + "</head>"))


def add_loader(out: Path) -> None:
    """Inline the loading screen in both pages and let the worker report its downloads.

    The screen is plain HTML and CSS at the top of <body>, so it shows before any script
    runs. Its progress bar counts the bytes the Pyodide worker fetches (reported by
    worker_hook.js) against the size of everything in pyodide/ and public/.
    """
    total = sum(
        f.stat().st_size
        for d in ("pyodide", "public")
        if (out / d).exists()
        for f in (out / d).iterdir()
        # pyodide.asm.mjs is imported as a module, not fetched, so the hook never sees it.
        if f.is_file() and f.name != "pyodide.asm.mjs"
    )
    preview = base64.b64encode((LOADER / "loader-preview.webp").read_bytes()).decode()
    loader = (
        (LOADER / "loader.html")
        .read_text()
        .replace("__PREVIEW__", f"data:image/webp;base64,{preview}")
        .replace("__TOTAL_BYTES__", str(total))
        .replace("__REPO__", REPO_URL)
    )
    for index in (out / "index.html", out / "m" / "index.html"):
        html = index.read_text()
        if html.count("<body>") != 1:
            raise SystemExit(f"no unique <body> in {index}")
        # marimo's stylesheets (about 400 kB) would hold back the first paint; after the
        # loader they only block what follows it, which the loader covers anyway.
        sheets = re.findall(r'<link rel="stylesheet"[^>]*>\n?', html)
        for sheet in sheets:
            html = html.replace(sheet, "", 1)
        index.write_text(html.replace("<body>", "<body>\n" + loader + "".join(sheets)))
    (worker,) = [f for f in (out / "assets").glob("worker-*.js")]
    worker.write_text((LOADER / "worker_hook.js").read_text() + worker.read_text())


def main(out: Path, cdn: bool) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / "stage"
        public = stage / "public"
        public.mkdir(parents=True)
        subprocess.run(
            ["uv", "build", "--wheel", "--out-dir", str(Path(tmp) / "dist"), str(ROOT)],
            check=True,
        )
        shutil.copy(Path(tmp) / "dist" / WHEEL, public / WHEEL)
        shutil.copy(ROOT / "data" / "presets" / "spx_fits.npz", public / "spx_fits.npz")
        shutil.copy(
            ROOT / "data" / "presets" / "spx_model_fits.json", public / "spx_model_fits.json"
        )
        plotly = locked_wheel("plotly", public)
        slim_plotly(public / plotly)
        for name in ("vol_lab.py", "vol_lab_mobile.py"):
            notebook = (ROOT / "dashboard" / name).read_text()
            for filename in (WHEEL, plotly):
                if filename not in notebook:
                    raise SystemExit(f"dashboard/{name} does not install {filename}")
            (stage / name).write_text(notebook)
        built = Path(tmp) / "built"
        export(stage / "vol_lab.py", built)
        # The phone page lives in m/ and shares assets/, public/ and pyodide/ with the
        # desktop page (marimo emits the same asset files for both).
        mobile = Path(tmp) / "mobile"
        mobile.mkdir()
        shutil.copy(stage / "vol_lab_mobile.py", mobile / "vol_lab_mobile.py")
        export(mobile / "vol_lab_mobile.py", mobile / "out")
        html = (mobile / "out" / "index.html").read_text()
        refs = set(re.findall(r'(?:href|src)="\./([^"]+)"', html))
        missing = [r for r in refs if not (built / r).exists()]
        if missing:
            raise SystemExit(f"mobile page needs files the desktop export lacks: {missing[:5]}")
        (built / "m").mkdir()
        (built / "m" / "index.html").write_text(re.sub(r'((?:href|src)=")\./', r"\1../", html))
        # marimo's runtime also asks for favicon.ico next to the page.
        shutil.copy(built / "favicon.ico", built / "m" / "favicon.ico")
        # The redirect goes straight after the viewport tag (before it, phones report a
        # 980 px layout width) and ahead of anything that could delay it.
        desktop = built / "index.html"
        html = desktop.read_text()
        (viewport,) = re.findall(r'<meta name="viewport"[^>]*>', html)
        desktop.write_text(html.replace(viewport, viewport + "\n" + DESKTOP_HEAD, 1))
        add_head(built / "m" / "index.html", MOBILE_HEAD.replace("MOBILE_CSS", MOBILE_CSS))
        # marimo's export adds an assistant prompt file that a public site does not need.
        (built / "CLAUDE.md").unlink(missing_ok=True)
        if not cdn:
            self_host(built)
        add_loader(built)
        # Refill OUTPUT_DIR in place (a server already serving it keeps working).
        out.mkdir(parents=True, exist_ok=True)
        for child in out.iterdir():
            shutil.rmtree(child) if child.is_dir() else child.unlink()
        for child in built.iterdir():
            shutil.move(str(child), out / child.name)
    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"wrote {out} ({total / 1e6:.1f} MB)")


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) not in (1, 2) or (len(args) == 2 and args[1] != "--cdn"):
        raise SystemExit("usage: python scripts/build_public_vollab.py OUTPUT_DIR [--cdn]")
    main(Path(args[0]).resolve(), cdn=len(args) == 2)
