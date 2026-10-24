"""Static checks on custom strategy code (a second line of defence; the container is the boundary).

``check_source(code)`` parses the code and rejects anything outside a narrow shape:

* size: at most ``MAX_BYTES`` bytes, ``MAX_LINES`` lines, ``MAX_NODES`` AST nodes;
* top level: imports, ``def``/``class``, plain assignments, a docstring; exactly one
  ``class Strategy`` with no bases, keywords or decorators, defining
  ``on_bar(self, view, ctx)`` and optionally ``setup(self, ctx)``;
* imports: only modules in ``ALLOWED_MODULES``, absolute, undotted, no ``*``;
* no name or attribute starting with ``_`` (dunders and privates), no frame/code
  object attributes (``f_*``, ``co_*``, ``gi_*``, ...), no name in ``DENIED_NAMES``
  (``eval``, ``exec``, ``open``, ``getattr`` ...) and no attribute in
  ``DENIED_ATTRS`` (``format``, numpy file I/O, ``random`` ...);
* no string constant containing ``__`` (closes eval-a-string tricks such as
  annotation evaluation);
* no ``global``, ``async``, ``await``, ``yield``, ``match``.

The same function runs on the host (broker, ``custom_check`` stage) and again
inside the container before the code is executed. Stdlib only.
"""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass

MAX_BYTES = 8 * 1024
MAX_LINES = 200
MAX_NODES = 4000

# pp_ta (the registry's indicator code) joins this list once the registry branch lands.
ALLOWED_MODULES = frozenset(
    {"math", "statistics", "collections", "itertools", "functools", "dataclasses", "typing", "numpy"}
)

DENIED_NAMES = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "open",
        "input",
        "breakpoint",
        "help",
        "getattr",
        "setattr",
        "delattr",
        "hasattr",
        "globals",
        "locals",
        "vars",
        "dir",
        "type",
        "super",
        "id",
        "memoryview",
        "print",
        "exit",
        "quit",
        "get_type_hints",
        "ForwardRef",
    }
)

DENIED_ATTRS = frozenset(
    {
        # string formatting can walk attributes ("{0.__class__}".format(x))
        "format",
        "format_map",
        # introspection
        "mro",
        "get_type_hints",
        "ForwardRef",
        "inspect",
        # modules reachable as attributes of allowed modules
        "sys",
        "os",
        "builtins",
        "modules",
        "subprocess",
        "importlib",
        "ctypes",
        "ctypeslib",
        "lib",
        "testing",
        "f2py",
        "distutils",
        # numpy file and process access
        "load",
        "loads",
        "save",
        "savez",
        "savez_compressed",
        "savetxt",
        "loadtxt",
        "genfromtxt",
        "fromfile",
        "fromregex",
        "tofile",
        "dump",
        "dumps",
        "memmap",
        "DataSource",
        "show_config",
        "show_runtime",
        "info",
        "source",
        "seterrcall",
        # unseeded global RNG (numpy.random, statistics.random) breaks determinism
        "random",
    }
)

DENIED_ATTR_PREFIXES = ("_", "f_", "co_", "gi_", "cr_", "ag_", "tb_")

ALLOWED_TOP_LEVEL = (ast.Import, ast.ImportFrom, ast.ClassDef, ast.FunctionDef, ast.Assign, ast.AnnAssign)
DENIED_NODES = (
    ast.Global,
    ast.AsyncFunctionDef,
    ast.AsyncFor,
    ast.AsyncWith,
    ast.Await,
    ast.Yield,
    ast.YieldFrom,
    ast.Match,  # class patterns match attributes by name strings (case C(__class__=x))
)


@dataclass(frozen=True)
class StaticReport:
    ok: bool
    errors: tuple[str, ...]
    code_sha256: str
    bytes: int
    lines: int

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "errors": list(self.errors),
            "code_sha256": self.code_sha256,
            "bytes": self.bytes,
            "lines": self.lines,
        }


def _where(node: ast.AST) -> str:
    return f"line {getattr(node, 'lineno', '?')}"


def _check_strategy_class(node: ast.ClassDef, errors: list[str]) -> None:
    if node.bases or node.keywords or node.decorator_list:
        errors.append(f"{_where(node)}: class Strategy takes no bases, keywords or decorators")
    methods = {item.name: item for item in node.body if isinstance(item, ast.FunctionDef)}
    expected = {"on_bar": ["self", "view", "ctx"], "setup": ["self", "ctx"]}
    if "on_bar" not in methods:
        errors.append(f"{_where(node)}: class Strategy must define on_bar(self, view, ctx)")
    for name, params in expected.items():
        fn = methods.get(name)
        if fn is None:
            continue
        args = fn.args
        got = [a.arg for a in args.posonlyargs + args.args]
        if got != params or args.vararg or args.kwarg or args.kwonlyargs or fn.decorator_list:
            errors.append(f"{_where(fn)}: {name} must be exactly def {name}({', '.join(params)})")


def _check_node(node: ast.AST, errors: list[str]) -> None:
    if isinstance(node, DENIED_NODES):
        errors.append(f"{_where(node)}: {node.__class__.__name__} is not allowed")
    elif isinstance(node, ast.Import):
        for alias in node.names:
            if alias.name not in ALLOWED_MODULES:
                errors.append(f"{_where(node)}: import of {alias.name!r} is not allowed")
            if alias.asname and alias.asname.startswith("_"):
                errors.append(f"{_where(node)}: alias {alias.asname!r} is not allowed")
    elif isinstance(node, ast.ImportFrom):
        if node.level or node.module not in ALLOWED_MODULES:
            errors.append(f"{_where(node)}: import from {node.module!r} is not allowed")
        for alias in node.names:
            name = alias.name
            if name == "*" or name.startswith("_") or name in DENIED_NAMES or name in DENIED_ATTRS:
                errors.append(f"{_where(node)}: importing {name!r} is not allowed")
    elif isinstance(node, ast.Name):
        if node.id.startswith("_") or node.id in DENIED_NAMES:
            errors.append(f"{_where(node)}: name {node.id!r} is not allowed")
    elif isinstance(node, ast.Attribute):
        if node.attr in DENIED_ATTRS or node.attr.startswith(DENIED_ATTR_PREFIXES):
            errors.append(f"{_where(node)}: attribute {node.attr!r} is not allowed")
    elif isinstance(node, ast.FunctionDef | ast.ClassDef):
        if node.name.startswith("_") or node.name in DENIED_NAMES:
            errors.append(f"{_where(node)}: definition name {node.name!r} is not allowed")
    elif isinstance(node, ast.arg):
        if node.arg.startswith("_"):
            errors.append(f"{_where(node)}: argument name {node.arg!r} is not allowed")
    elif isinstance(node, ast.keyword):
        if node.arg is not None and node.arg.startswith("_"):
            errors.append(f"{_where(node)}: keyword {node.arg!r} is not allowed")
    elif isinstance(node, ast.ExceptHandler):
        if node.name and node.name.startswith("_"):
            errors.append(f"{_where(node)}: name {node.name!r} is not allowed")
    elif isinstance(node, ast.Constant):
        value = node.value
        if (isinstance(value, str) and "__" in value) or (isinstance(value, bytes) and b"__" in value):
            errors.append(f"{_where(node)}: string constants may not contain '__'")


def check_source(code: str) -> StaticReport:
    raw = code.encode("utf-8", errors="surrogatepass") if isinstance(code, str) else b""
    lines = code.count("\n") + 1 if isinstance(code, str) and code else 0
    sha = hashlib.sha256(raw).hexdigest()
    errors: list[str] = []

    def report() -> StaticReport:
        return StaticReport(not errors, tuple(errors), sha, len(raw), lines)

    if not isinstance(code, str) or not code.strip():
        errors.append("code must be a non-empty string")
        return report()
    if len(raw) > MAX_BYTES:
        errors.append(f"code is {len(raw)} bytes; the limit is {MAX_BYTES}")
    if lines > MAX_LINES:
        errors.append(f"code is {lines} lines; the limit is {MAX_LINES}")
    if errors:
        return report()
    try:
        tree = ast.parse(code, filename="<strategy>", mode="exec")
    except (SyntaxError, ValueError) as exc:
        errors.append(f"syntax error: {exc.__class__.__name__}: {getattr(exc, 'msg', exc)}")
        return report()

    nodes = list(ast.walk(tree))
    if len(nodes) > MAX_NODES:
        errors.append(f"code has {len(nodes)} AST nodes; the limit is {MAX_NODES}")
        return report()

    strategies = []
    for index, stmt in enumerate(tree.body):
        is_docstring = (
            index == 0
            and isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str)
        )
        if not (is_docstring or isinstance(stmt, ALLOWED_TOP_LEVEL)):
            errors.append(f"{_where(stmt)}: only imports, definitions and assignments are allowed at top level")
        if isinstance(stmt, ast.ClassDef) and stmt.name == "Strategy":
            strategies.append(stmt)
    if len(strategies) != 1:
        errors.append("code must define exactly one top-level class Strategy")
    else:
        _check_strategy_class(strategies[0], errors)

    for node in nodes:
        _check_node(node, errors)
    return report()
