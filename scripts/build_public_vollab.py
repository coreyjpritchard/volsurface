"""Build the static web version of the vol lab: one folder, no server, no account.

Run from the repository root:

    python scripts/build_public_vollab.py OUTPUT_DIR [--cdn]

The folder runs ``dashboard/vol_lab.py`` in the visitor's browser with Pyodide
(``marimo export html-wasm --mode run``). Next to the page, in ``public/``, it ships
the pure-Python ``volsurface`` wheel built from this checkout and the SPX fits in
``data/presets/spx_fits.npz`` (14 October 2026 SPXW snapshot), plus the plotly wheel
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

import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WHEEL = "volsurface-0.1.0-py3-none-any.whl"
CACHE = Path.home() / ".cache" / "volsurface-web"
# Packages the page imports: marimo's boot set, what marimo loads for run mode, and ours.
RUNTIME_PACKAGES = [
    "micropip",
    "msgspec",
    "marimo-base",
    "markdown",
    "pymdown-extensions",
    "narwhals",
    "packaging",
    "pygments",
    "docutils",
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
    index = out / "index.html"
    html = index.read_text()
    tag = '<marimo-wasm hidden="">'
    if html.count(tag) != 1:
        raise SystemExit("could not find the marimo-wasm element in index.html")
    index.write_text(
        html.replace(
            tag,
            '<marimo-wasm hidden="" data-pyodide-index-url="./pyodide/" '
            'data-pyodide-lockfile-url="./pyodide/pyodide-lock.json">',
        )
    )
    mathjax = {
        u
        for t in text.values()
        for u in re.findall(r"https://cdn\.jsdelivr\.net/npm/mathjax[^`'\"]+", t)
    }
    for url in mathjax:
        local = "vendor/mathjax/" + url.rsplit("/", 1)[1]
        (out / local).parent.mkdir(parents=True, exist_ok=True)
        (out / local).write_bytes(fetch(url))
        for f, t in text.items():
            if url in t:
                text[f] = t = t.replace(url, "./" + local)
                f.write_text(t)


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
        notebook = (ROOT / "dashboard" / "vol_lab.py").read_text()
        for filename in (WHEEL, locked_wheel("plotly", public)):
            if filename not in notebook:
                raise SystemExit(f"dashboard/vol_lab.py does not install {filename}")
        (stage / "vol_lab.py").write_text(notebook)
        if out.exists():
            shutil.rmtree(out)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "marimo",
                "export",
                "html-wasm",
                str(stage / "vol_lab.py"),
                "-o",
                str(out),
                "--mode",
                "run",
                "--no-show-code",
            ],
            check=True,
        )
    # marimo's export adds an assistant prompt file that a public site does not need.
    (out / "CLAUDE.md").unlink(missing_ok=True)
    if not cdn:
        self_host(out)
    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"wrote {out} ({total / 1e6:.1f} MB)")


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) not in (1, 2) or (len(args) == 2 and args[1] != "--cdn"):
        raise SystemExit("usage: python scripts/build_public_vollab.py OUTPUT_DIR [--cdn]")
    main(Path(args[0]).resolve(), cdn=len(args) == 2)
