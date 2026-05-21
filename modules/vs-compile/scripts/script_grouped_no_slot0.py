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
# Adds subgraph/cluster support on top of the basic Digraph used by script.py.
# ---------------------------------------------------------------------------

_ID_SAFE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$|^-?\d+(?:\.\d+)?$")


def _is_html_label(s):
    s = str(s)
    return s.startswith("<") and s.endswith(">")


def _quote_val(v):
    s = str(v)
    if _is_html_label(s):
        return s
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


class _Block:
    """A container that can hold attributes, default styles, nodes, edges, and nested subgraphs."""

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
        self._statements.append(f"{_quote_id(name)} [{_fmt_attrs(attrs)}];")

    def edge(self, tail, head, label=None, **style):
        attrs = {}
        if label is not None:
            attrs["label"] = label
        attrs.update(style)
        self._statements.append(
            f"{_quote_endpoint(tail)} -> {_quote_endpoint(head)} [{_fmt_attrs(attrs)}];"
        )

    def subgraph(self, name):
        sg = _Block()
        self._statements.append(("subgraph", name, sg))
        return sg

    def _emit_body(self, indent):
        pad = "  " * indent
        out = []
        for k, v in self._graph_attrs.items():
            out.append(f"{pad}{k}={_quote_val(v)};")
        if self._default_node_attrs:
            out.append(f"{pad}node [{_fmt_attrs(self._default_node_attrs)}];")
        if self._default_edge_attrs:
            out.append(f"{pad}edge [{_fmt_attrs(self._default_edge_attrs)}];")
        for stmt in self._statements:
            if isinstance(stmt, tuple) and stmt[0] == "subgraph":
                _, name, sg = stmt
                out.append(f"{pad}subgraph {_quote_id(name)} {{")
                out.extend(sg._emit_body(indent + 1))
                out.append(f"{pad}}}")
            else:
                out.append(f"{pad}{stmt}")
        return out


class Digraph(_Block):
    def source(self):
        out = ["digraph G {"]
        out.extend(self._emit_body(1))
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


def slot_sort_key(slot):
    if slot is None:
        return (1, 0, "")
    if isinstance(slot, int):
        return (0, slot, "")
    s = str(slot)
    if s.lstrip("-").isdigit():
        return (0, int(s), "")
    return (0, 0, s)


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

    def is_slot_zero(name):
        return node_attrs.get(name, {}).get("slot") == 0

    visible_nodes = {n for n in nodes if not is_slot_zero(n)}

    node_reps = {n: parse_reps(b) for n, b in node_bodies.items() if n in visible_nodes}

    cstrs = []
    for attrs, _, c_start, _ in find_all_ops(body, "cstr"):
        if any(s <= c_start < e for s, e in rop_spans):
            continue
        cstrs.append(attrs)

    node_indices = {n: set() for n in visible_nodes}
    edges = []
    for c in cstrs:
        a = c.get("src")
        b = c.get("dst")
        if a not in visible_nodes or b not in visible_nodes:
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
    dot.attr(label=epoch_name, labelloc="t", nodesep="0.6", ranksep="0.8",
             compound="true", rankdir="TB", newrank="true")
    dot.attr("node", shape="record")

    slots = {}
    for n in visible_nodes:
        slot = node_attrs.get(n, {}).get("slot")
        slots.setdefault(slot, []).append(n)

    sorted_slot_keys = sorted(slots.keys(), key=slot_sort_key)
    ordered_members = [sorted(slots[s]) for s in sorted_slot_keys]
    max_count = max((len(m) for m in ordered_members), default=0)

    def emit_node(target, n):
        indices = sorted(node_indices[n], key=sort_key)
        reps = node_reps.get(n, [])
        rep_section = ""
        if reps:
            rep_boxes = "|".join(format_rep(r) for r in reps)
            rep_section = f"|{{{{{rep_boxes}}}}}"
        slot = node_attrs.get(n, {}).get("slot")
        title = f"{n} (slot={slot})" if slot is not None else n
        head = f"{title}{rep_section}"
        if indices:
            ports = "|".join(f"<{port_name(i)}>{idx_label(i)}" for i in indices)
            target.node(n, label=f"{{{head}|{{{ports}}}}}")
        elif rep_section:
            target.node(n, label=f"{{{head}}}")
        else:
            target.node(n, label=title)

    for slot, members in zip(sorted_slot_keys, ordered_members):
        slot_label = f"slot {slot}" if slot is not None else "slot ?"
        cluster_id = f"cluster_slot_{slot if slot is not None else 'none'}"
        sg = dot.subgraph(cluster_id)
        sg.attr(label=slot_label, style="rounded,filled", color="gray70",
                fillcolor="gray95", margin="30")
        for n in members:
            emit_node(sg, n)
        pad_names = []
        for k in range(max_count - len(members)):
            pad_name = f"__pad_{cluster_id}_{k}"
            sg.node(pad_name, label="", style="invis", shape="box",
                    width="2.0", height="0.4", fixedsize="true")
            pad_names.append(pad_name)
        column = members + pad_names
        # Stack instructions inside this slot vertically.
        for k in range(len(column) - 1):
            sg.edge(column[k], column[k + 1], style="invis",
                    constraint="true", weight="1000")

    legend = (
        '<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">'
        '<TR><TD COLSPAN="2"><B>Legend</B></TD></TR>'
        '<TR><TD BGCOLOR="darkgreen" WIDTH="20"></TD><TD ALIGN="LEFT">RAW-dependency</TD></TR>'
        '<TR><TD BGCOLOR="red" WIDTH="20"></TD><TD ALIGN="LEFT">WAR-dependency</TD></TR>'
        '<TR><TD ALIGN="CENTER"><FONT FACE="monospace">- - -</FONT></TD><TD ALIGN="LEFT">not-equal dependency</TD></TR>'
        '</TABLE>>'
    )
    dot.node("__legend__", label=legend, shape="plaintext")

    # When several edges share the same (src, dst) node pair, give each a
    # distinct color from a palette so parallel edges are easy to tell apart.
    palette = ["red", "darkred", "firebrick", "crimson", "indianred",
               "tomato", "orangered", "brown", "maroon", "salmon",
               "lightcoral", "palevioletred"]
    pair_indices = {}
    for ei, (a, _, b, _, _, _, _) in enumerate(edges):
        pair_indices.setdefault((a, b), []).append(ei)
    override_color = {}
    for pair, idxs in pair_indices.items():
        if len(idxs) > 1:
            for pos, ei in enumerate(idxs):
                override_color[ei] = palette[pos % len(palette)]

    for ei, (a, a_idx, b, b_idx, mn, mx, is_neq) in enumerate(edges):
        tail = endpoint(a, a_idx)
        head = endpoint(b, b_idx)
        forced = override_color.get(ei)
        if is_neq:
            label = f"!=[{mn}]"
            color = forced if forced else "red"
            dot.edge(tail, head, label=label, color=color, fontcolor=color, style="dashed")
        elif mn == mx:
            label = f"[{mn},{mn}]"
            color = forced if forced else "red"
            extra = {"dir": "both"} if mn == 0 else {}
            dot.edge(tail, head, label=label, color=color, fontcolor=color, **extra)
        elif mx is None or mx >= INF_DELAY:
            label = f"[{mn},inf)"
            color = forced if forced else "darkgreen"
            dot.edge(tail, head, label=label, color=color, fontcolor=color)
        else:
            label = f"[{mn},{mx}]"
            color = forced if forced else "red"
            dot.edge(tail, head, label=label, color=color, fontcolor=color)

    return dot


epochs = find_epochs(src)
if not epochs:
    epochs = [("graph", src)]

base = os.path.splitext(os.path.basename(MLIR_FILE))[0]
multi = len(epochs) > 1
for name, body in epochs:
    dot = build_graph(name, body)
    out_name = f"{base}_{name}_grouped_no_slot0" if multi else f"{base}_grouped_no_slot0"
    dot.render(out_name, format="png", view=False, cleanup=False)
