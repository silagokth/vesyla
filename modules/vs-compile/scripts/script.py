import os
import re
import subprocess
import sys

if len(sys.argv) < 2:
    sys.exit(f"usage: {sys.argv[0]} <mlir-file>")
MLIR_FILE = sys.argv[1]
INF_DELAY = 10000000  # max_delay value treated as +inf


# ---------------------------------------------------------------------------
# Minimal Graphviz emitter — only the system `dot` binary is required.
# Mirrors the subset of `graphviz.Digraph` that this script uses.
# ---------------------------------------------------------------------------

_ID_SAFE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$|^-?\d+(?:\.\d+)?$")


def _is_html_label(s):
    s = str(s)
    return s.startswith("<") and s.endswith(">")


def _quote_val(v):
    s = str(v)
    if _is_html_label(s):
        return s  # raw HTML label — emitted unquoted, wrapped in <...>
    return '"' + s.replace('"', r'\"') + '"'


def _fmt_attrs(attrs):
    return ", ".join(f"{k}={_quote_val(v)}" for k, v in attrs.items())


def _quote_id(name):
    s = str(name)
    if _ID_SAFE.match(s):
        return s
    return '"' + s.replace('"', r'\"') + '"'


def _quote_endpoint(ep):
    s = str(ep)
    if ":" in s:
        node, port = s.split(":", 1)
        return f"{_quote_id(node)}:{port}"
    return _quote_id(s)


class Digraph:
    def __init__(self):
        self._graph_attrs = {}
        self._default_node_attrs = {}
        self._default_edge_attrs = {}
        self._statements = []

    def attr(self, target=None, **kwargs):
        if target in (None, "graph"):
            self._graph_attrs.update(kwargs)
        elif target == "node":
            self._default_node_attrs.update(kwargs)
        elif target == "edge":
            self._default_edge_attrs.update(kwargs)

    def node(self, name, label=None, **style):
        attrs = {}
        if label is not None:
            attrs["label"] = label
        attrs.update(style)
        self._statements.append(f"  {_quote_id(name)} [{_fmt_attrs(attrs)}];")

    def edge(self, tail, head, label=None, **style):
        attrs = {}
        if label is not None:
            attrs["label"] = label
        attrs.update(style)
        self._statements.append(
            f"  {_quote_endpoint(tail)} -> {_quote_endpoint(head)} [{_fmt_attrs(attrs)}];"
        )

    def source(self):
        out = ["digraph G {"]
        for k, v in self._graph_attrs.items():
            out.append(f"  {k}={_quote_val(v)};")
        if self._default_node_attrs:
            out.append(f"  node [{_fmt_attrs(self._default_node_attrs)}];")
        if self._default_edge_attrs:
            out.append(f"  edge [{_fmt_attrs(self._default_edge_attrs)}];")
        out.extend(self._statements)
        out.append("}")
        return "\n".join(out) + "\n"

    def render(self, name, format="png", view=False, cleanup=True):
        dot_path = f"{name}.dot"
        out_path = f"{name}.{format}"
        with open(dot_path, "w") as f:
            f.write(self.source())
        try:
            subprocess.run(
                ["dot", f"-T{format}", "-o", out_path, dot_path], check=True
            )
        finally:
            if cleanup:
                try:
                    os.unlink(dot_path)
                except OSError:
                    pass
        if view:
            self._open(out_path)
        return out_path

    @staticmethod
    def _open(path):
        try:
            if sys.platform.startswith("linux"):
                subprocess.Popen(
                    ["xdg-open", path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
            elif sys.platform == "darwin":
                subprocess.Popen(["open", path], start_new_session=True)
            elif sys.platform == "win32":
                os.startfile(path)
        except FileNotFoundError:
            pass

src = open(MLIR_FILE).read()


# ---------------------------------------------------------------------------
# Generic MLIR attr-dict parsing
# ---------------------------------------------------------------------------

def _balance(text, i, open_c="{", close_c="}"):
    depth = 0
    while i < len(text):
        c = text[i]
        if c == open_c:
            depth += 1
        elif c == close_c:
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return -1


def parse_typed_attr(text, i):
    nm = re.match(r"#[\w.]+", text[i:])
    name = nm.group(0)
    i += nm.end()
    if i >= len(text) or text[i] != "<":
        return {"_type": name}, i
    j = _balance(text, i, "<", ">")
    inner = text[i + 1:j - 1]
    if name == "#pasm.anchor_range":
        return _parse_anchor_range_inner(inner, name), j
    if name == "#pasm.delay":
        return _parse_delay_inner(inner, name), j
    params = {}
    for pm in re.finditer(r"(\w+)\s*=\s*(-?\d+)", inner):
        params[pm.group(1)] = int(pm.group(2))
    return {"_type": name, **params}, j


def _parse_anchor_range_inner(inner, name):
    out = {"_type": name, "instr": None, "event": "",
           "idx_lo": [], "idx_hi": []}
    k = 0
    while k < len(inner) and inner[k] in " \t\n":
        k += 1
    mm = re.match(r"@([\w]+)", inner[k:])
    if mm:
        out["instr"] = mm.group(1)
        k += mm.end()
    while k < len(inner) and inner[k] in " \t\n,":
        k += 1
    if k < len(inner) and inner[k] == '"':
        end = inner.find('"', k + 1)
        if end != -1:
            out["event"] = inner[k + 1:end]
            k = end + 1
    while k < len(inner):
        while k < len(inner) and inner[k] in " \t\n,":
            k += 1
        if k >= len(inner) or inner[k] != "[":
            break
        end = inner.find("]", k)
        if end == -1:
            break
        rng = inner[k + 1:end].strip()
        if ":" in rng:
            lo_s, hi_s = rng.split(":", 1)
            lo, hi = int(lo_s.strip()), int(hi_s.strip())
        else:
            lo = int(rng)
            hi = lo
        out["idx_lo"].append(lo)
        out["idx_hi"].append(hi)
        k = end + 1
    return out


def _parse_delay_inner(inner, name):
    out = {"_type": name, "min": None, "max": None}
    mm = re.search(r"\[\s*(-?\d+)?\s*,\s*(-?\d+)?\s*\]", inner)
    if mm:
        if mm.group(1) is not None:
            out["min"] = int(mm.group(1))
        if mm.group(2) is not None:
            out["max"] = int(mm.group(2))
    return out


def unwrap_cstr_attrs(c):
    src = c.get("src")
    if isinstance(src, dict):
        c["src_event"] = src.get("event", "") or ""
        c["src_idx_lo"] = src.get("idx_lo", []) or []
        c["src_idx_hi"] = src.get("idx_hi", []) or []
        c["src"] = src.get("instr")
    dst = c.get("dst")
    if isinstance(dst, dict):
        c["dst_event"] = dst.get("event", "") or ""
        c["dst_idx_lo"] = dst.get("idx_lo", []) or []
        c["dst_idx_hi"] = dst.get("idx_hi", []) or []
        c["dst"] = dst.get("instr")
    delay = c.get("delay")
    if isinstance(delay, dict):
        if delay.get("min") is not None:
            c["min_delay"] = delay["min"]
        if delay.get("max") is not None:
            c["max_delay"] = delay["max"]


def parse_attr_dict(s):
    s = s.strip()
    assert s.startswith("{") and s.endswith("}"), s[:40]
    body = s[1:-1]
    out = {}
    i = 0
    while i < len(body):
        while i < len(body) and body[i] in " \t\n,":
            i += 1
        if i >= len(body):
            break
        m = re.match(r"(\w+)\s*=\s*", body[i:])
        if not m:
            break
        key = m.group(1)
        i += m.end()
        if body[i] == '"':
            j = body.find('"', i + 1)
            out[key] = body[i + 1:j]
            i = j + 1
        elif body[i] == "@":
            j = i + 1
            while j < len(body) and (body[j].isalnum() or body[j] == "_"):
                j += 1
            out[key] = body[i + 1:j]
            i = j
        elif body[i] == "{":
            j = _balance(body, i)
            out[key] = parse_attr_dict(body[i:j])
            i = j
        elif body[i] == "#":
            out[key], i = parse_typed_attr(body, i)
        elif body[i:i + 5] == "array":
            j = body.find(">", i)
            content = body[i:j + 1]
            colon = content.find(":")
            nums = re.findall(r"-?\d+", content[colon + 1:]) if colon != -1 else []
            out[key] = [int(n) for n in nums]
            i = j + 1
        elif re.match(r"(?:true|false)\b", body[i:]):
            tok = re.match(r"(?:true|false)\b", body[i:]).group(0)
            out[key] = (tok == "true")
            i += len(tok)
        else:
            num = re.match(r"-?\d+", body[i:])
            if num:
                out[key] = int(num.group(0))
                i += num.end()
                # skip optional ` : i32` etc.
                while i < len(body) and body[i] in " \t":
                    i += 1
                if i < len(body) and body[i] == ":":
                    i += 1
                    while i < len(body) and body[i] in " \t":
                        i += 1
                    tm = re.match(r"\w+", body[i:])
                    if tm:
                        i += tm.end()
            else:
                tok = re.match(r"\S+", body[i:])
                out[key] = tok.group(0) if tok else ""
                i += len(out[key])
    return out


def find_op(text, op_name, start=0):
    """Find the next `pasm.<op_name><` at or after `start`.
    Returns (attrs, body_or_None, end_index, match_start) or None."""
    pat = re.compile(rf"pasm\.{re.escape(op_name)}<\s*")
    m = pat.search(text, start)
    if not m:
        return None
    if m.end() >= len(text) or text[m.end()] != "{":
        return None
    attr_end = _balance(text, m.end())
    attrs = parse_attr_dict(text[m.end():attr_end])
    j = attr_end
    while j < len(text) and text[j].isspace():
        j += 1
    if j >= len(text) or text[j] != ">":
        return None
    j += 1
    while j < len(text) and text[j].isspace():
        j += 1
    if j < len(text) and text[j] == "{":
        body_end = _balance(text, j)
        return attrs, text[j + 1:body_end - 1], body_end, m.start()
    return attrs, None, j, m.start()


def find_all_ops(text, op_name):
    i = 0
    while True:
        r = find_op(text, op_name, i)
        if r is None:
            break
        attrs, body, end, start = r
        yield attrs, body, start, end
        i = end


# ---------------------------------------------------------------------------
# pasm-specific extraction
# ---------------------------------------------------------------------------

def find_epochs(text):
    return [(a.get("id", "epoch"), b or "") for (a, b, _, _) in find_all_ops(text, "epoch")]


def parse_reps(rop_body):
    reps = []
    for attrs, _, _, _ in find_all_ops(rop_body, "instr"):
        if attrs.get("type") == "rep":
            param = attrs.get("param", {}) or {}
            reps.append({
                "iter": param.get("iter"),
                "step": param.get("step"),
                "delay": param.get("delay"),
            })
    return reps


def decode_one_hot(s):
    s = str(s)
    try:
        if s.startswith(("0b", "0B")):
            v = int(s, 2)
        elif s.startswith(("0x", "0X")):
            v = int(s, 16)
        else:
            v = int(s)
    except ValueError:
        return s
    bits = []
    i = 0
    while v:
        if v & 1:
            bits.append(i)
        v >>= 1
        i += 1
    return "[" + ",".join(str(b) for b in bits) + "]"


def extract_st_ops(rop_body):
    groups = {}
    order = []
    for attrs, _, _, _ in find_all_ops(rop_body, "instr"):
        param = attrs.get("param", {}) or {}
        if "source" in param and "target" in param:
            opt = str(param.get("option", ""))
            target = param["target"]
            if attrs.get("type") == "route":
                target = decode_one_hot(target)
            else:
                target = str(target)
            source = str(param["source"])
            groups.setdefault(opt, []).append((source, target))
            if opt not in order:
                order.append(opt)
    return [(opt, groups[opt]) for opt in order]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def port_name(idx):
    if not idx:
        return None
    return "p_" + "_".join(str(i).replace(":", "_") for i in idx)


def idx_label(idx):
    return "".join(f"[{i}]" for i in idx)


def sort_key(idx):
    out = []
    for i in idx:
        if isinstance(i, int):
            out.append((0, i))
        elif isinstance(i, str) and i.lstrip("-").isdigit():
            out.append((0, int(i)))
        else:
            out.append((1, str(i)))
    return tuple(out)


def format_rep(rep):
    parts = []
    if rep.get("iter") is not None:
        parts.append(f"iter={rep['iter']}")
    if rep.get("step") is not None:
        parts.append(f"step={rep['step']}")
    if rep.get("delay") is not None:
        parts.append(f"delay={rep['delay']}")
    return "\\n".join(parts)


def endpoint(name, idx):
    p = port_name(idx)
    return f"{name}:{p}" if p else name


def cstr_idx(lo_arr, hi_arr):
    """Combine `_idx_lo` and `_idx_hi` arrays into a tuple of strings (`lo` if equal, `lo:hi` otherwise)."""
    lo_arr = lo_arr or []
    hi_arr = hi_arr or []
    n = max(len(lo_arr), len(hi_arr))
    out = []
    for k in range(n):
        lo = lo_arr[k] if k < len(lo_arr) else None
        hi = hi_arr[k] if k < len(hi_arr) else None
        if lo == hi:
            out.append(str(lo))
        else:
            out.append(f"{lo}:{hi}")
    return tuple(out)


def build_graph(epoch_name, body):
    nodes = set()
    node_attrs = {}
    node_bodies = {}
    rop_spans = []
    for attrs, rbody, start, end in find_all_ops(body, "rop"):
        sym = attrs.get("sym_name")
        if not sym:
            continue
        nodes.add(sym)
        node_attrs[sym] = attrs
        node_bodies[sym] = rbody or ""
        rop_spans.append((start, end))

    node_reps = {n: parse_reps(b) for n, b in node_bodies.items()}

    # Collect cstrs that are direct children of the epoch (not inside a rop body).
    cstrs = []
    for attrs, _, c_start, _ in find_all_ops(body, "cstr"):
        if any(s <= c_start < e for s, e in rop_spans):
            continue
        unwrap_cstr_attrs(attrs)
        cstrs.append(attrs)

    node_indices = {n: set() for n in nodes}
    edges = []
    for c in cstrs:
        a = c.get("src")
        b = c.get("dst")
        if a not in nodes or b not in nodes:
            continue
        a_idx = cstr_idx(c.get("src_idx_lo"), c.get("src_idx_hi"))
        b_idx = cstr_idx(c.get("dst_idx_lo"), c.get("dst_idx_hi"))
        if a_idx:
            node_indices[a].add(a_idx)
        if b_idx:
            node_indices[b].add(b_idx)
        mn = c.get("min_delay", 0)
        mx = c.get("max_delay")
        is_neq = bool(c.get("is_neq", False))
        edges.append((a, a_idx, b, b_idx, mn, mx, is_neq))

    dot = Digraph()
    dot.attr(label=epoch_name, labelloc="t", nodesep="0.6", ranksep="0.8")
    dot.attr("node", shape="record")

    for n in sorted(nodes):
        indices = sorted(node_indices[n], key=sort_key)
        reps = node_reps.get(n, [])
        rep_section = ""
        if reps:
            rep_boxes = "|".join(format_rep(r) for r in reps)
            rep_section = f"|{{{{{rep_boxes}}}}}"
        slot = node_attrs.get(n, {}).get("slot")
        slot_zero = slot == 0
        op_section = ""
        if slot_zero:
            st_groups = extract_st_ops(node_bodies[n])
            if st_groups:
                rows = []
                for opt, entries in st_groups:
                    entry_str = "|".join(f"{s}→{t}" for s, t in entries)
                    rows.append(f"{{option={opt}|{entry_str}}}")
                op_section = "|" + "|".join(rows)
        style = {"style": "filled", "fillcolor": "lightblue", "color": "blue"} if slot_zero else {}
        title = f"{n} (slot={slot})" if slot is not None else n
        head = f"{title}{rep_section}{op_section}"
        if indices:
            ports = "|".join(f"<{port_name(i)}>{idx_label(i)}" for i in indices)
            dot.node(n, label=f"{{{head}|{{{ports}}}}}", **style)
        elif rep_section or op_section:
            dot.node(n, label=f"{{{head}}}", **style)
        else:
            dot.node(n, label=title, **style)

    legend = (
        '<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">'
        '<TR><TD COLSPAN="2"><B>Legend</B></TD></TR>'
        '<TR><TD BGCOLOR="darkgreen" WIDTH="20"></TD><TD ALIGN="LEFT">RAW-dependency</TD></TR>'
        '<TR><TD BGCOLOR="red" WIDTH="20"></TD><TD ALIGN="LEFT">WAR-dependency</TD></TR>'
        '<TR><TD BGCOLOR="blue" WIDTH="20"></TD><TD ALIGN="LEFT">swb-dependency</TD></TR>'
        '<TR><TD ALIGN="CENTER"><FONT FACE="monospace">- - -</FONT></TD><TD ALIGN="LEFT">not-equal dependency</TD></TR>'
        '</TABLE>>'
    )
    dot.node("__legend__", label=legend, shape="plaintext")

    def is_slot_zero(name):
        return node_attrs.get(name, {}).get("slot") == 0

    def pick_color(src_name, dst_name, default):
        if is_slot_zero(src_name) or is_slot_zero(dst_name):
            return "blue"
        return default

    for a, a_idx, b, b_idx, mn, mx, is_neq in edges:
        if a not in nodes or b not in nodes:
            continue
        tail = endpoint(a, a_idx)
        head = endpoint(b, b_idx)
        if is_neq:
            label = f"!=[{mn}]"
            color = pick_color(a, b, "red")
            dot.edge(tail, head, label=label, color=color, fontcolor=color, style="dashed")
        elif mn == mx:
            label = f"[{mn},{mn}]"
            color = pick_color(a, b, "red")
            extra = {"dir": "both"} if mn == 0 else {}
            dot.edge(tail, head, label=label, color=color, fontcolor=color, **extra)
        elif mx is None or mx >= INF_DELAY:
            label = f"[{mn},inf)"
            color = pick_color(a, b, "darkgreen")
            dot.edge(tail, head, label=label, color=color, fontcolor=color)
        else:
            label = f"[{mn},{mx}]"
            color = pick_color(a, b, "red")
            dot.edge(tail, head, label=label, color=color, fontcolor=color)

    return dot


epochs = find_epochs(src)
if not epochs:
    epochs = [("graph", src)]

base = os.path.splitext(os.path.basename(MLIR_FILE))[0]
multi = len(epochs) > 1
for name, body in epochs:
    dot = build_graph(name, body)
    out_name = f"{base}_{name}" if multi else base
    dot.render(out_name, format="svg", view=False, cleanup=False)
    dot.render(out_name, format="png", view=False, cleanup=False)
