"""Architectural boundary, DAG layering, and static symbol checks across the repository.

Run:
    PYTHONPATH=src .venv/bin/python tests/test_architecture_imports.py
"""

from __future__ import annotations

import ast
import builtins
import py_compile
import re
import symtable
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILTIN_NAMES = set(dir(builtins)) | {"__annotations__", "__file__", "__name__"}

# Strict acyclic dependency rules among `cosmicstar.worldmodel.<pkg>` subpackages
ALLOWED_WORLDMODEL_DEPS: dict[str, set[str]] = {
    "core": set(),
    "nn": set(),
    "data": {"core"},
    "l0_physics": {"core", "nn", "data"},
    "l1_tactical": {"core", "nn", "data"},
    "l2_observer": {"core", "nn", "data", "l1_tactical"},
    "inspector": {"core", "nn", "data", "l0_physics", "l1_tactical", "l2_observer"},
}


def test_all_python_compiles_with_zero_undefined_globals() -> None:
    py_files = sorted(
        list((ROOT / "src").rglob("*.py"))
        + list((ROOT / "scripts").rglob("*.py"))
        + list((ROOT / "tests").rglob("*.py"))
    )
    assert py_files, "No Python files discovered"
    for p in py_files:
        py_compile.compile(str(p), doraise=True)
        src = p.read_text()
        st = symtable.symtable(src, str(p), "exec")
        top_defs = {
            s.get_name() for s in st.get_symbols() if s.is_assigned() or s.is_imported()
        }
        missing: set[str] = set()

        def walk(tbl):
            for s in tbl.get_symbols():
                if s.is_global() and not s.is_assigned() and not s.is_imported():
                    nm = s.get_name()
                    if nm not in top_defs and nm not in BUILTIN_NAMES:
                        missing.add(nm)
            for child in tbl.get_children():
                walk(child)

        walk(st)
        assert not missing, f"Undefined globals in {p.relative_to(ROOT)}: {sorted(missing)}"


def test_no_inverted_or_cross_script_imports() -> None:
    script_stems = {p.stem for p in (ROOT / "scripts").rglob("*.py")}
    for p in sorted(
        list((ROOT / "src").rglob("*.py"))
        + list((ROOT / "scripts").rglob("*.py"))
        + list((ROOT / "tests").rglob("*.py"))
    ):
        tree = ast.parse(p.read_text(), filename=str(p))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    assert top not in script_stems and top != "scripts", (
                        f"{p.relative_to(ROOT)} imports script '{alias.name}'"
                    )
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                top = node.module.split(".")[0]
                assert top not in script_stems and top != "scripts", (
                    f"{p.relative_to(ROOT)} imports from script '{node.module}'"
                )


def test_worldmodel_subpackage_dag() -> None:
    wm_root = ROOT / "src" / "cosmicstar" / "worldmodel"
    for pkg, allowed in ALLOWED_WORLDMODEL_DEPS.items():
        pkg_dir = wm_root / pkg
        assert pkg_dir.is_dir(), f"Missing expected subpackage {pkg_dir}"
        for p in sorted(pkg_dir.glob("*.py")):
            tree = ast.parse(p.read_text(), filename=str(p))
            for node in tree.body:
                for sub in ast.walk(node):
                    # Allow lazy CLI builder import inside core/vocab.py:build()
                    if pkg == "core" and p.name == "vocab.py" and isinstance(
                        node, ast.FunctionDef
                    ) and node.name == "build":
                        continue
                    target_pkg = None
                    if isinstance(sub, ast.ImportFrom):
                        if sub.level == 2:
                            if sub.module:
                                target_pkg = sub.module.split(".")[0]
                        elif sub.level == 0 and sub.module and sub.module.startswith(
                            "cosmicstar.worldmodel."
                        ):
                            target_pkg = sub.module.split(".")[2]
                    if target_pkg and target_pkg != pkg and target_pkg in ALLOWED_WORLDMODEL_DEPS:
                        assert target_pkg in allowed, (
                            f"Layer violation: {p.relative_to(ROOT)} imports '{target_pkg}' "
                            f"(allowed for '{pkg}': {sorted(allowed)})"
                        )


def test_inspector_ui_static_assets() -> None:
    ui_dir = ROOT / "src" / "cosmicstar" / "worldmodel" / "inspector" / "ui"
    index_html = (ui_dir / "index.html").read_text()
    for rel in re.findall(r'(?:href|src)="/ui/([^"]+)"', index_html):
        target = ui_dir / rel
        assert target.is_file(), f"Missing static UI file referenced by index.html: {target}"
        assert target.stat().st_size > 0, f"Empty static UI file: {target}"


if __name__ == "__main__":
    test_all_python_compiles_with_zero_undefined_globals()
    test_no_inverted_or_cross_script_imports()
    test_worldmodel_subpackage_dag()
    test_inspector_ui_static_assets()
    print("All architectural boundary, symtable, and UI asset checks passed!")
