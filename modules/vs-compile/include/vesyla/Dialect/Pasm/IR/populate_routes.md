# `populate_routes(graph, epoch, kind)`

Builds the routing-dependency graph for one `epoch` and one `kind`
("bulk" / "word"). Graph already contains start (`0, First`) and end
(`0, Last`) sentinels from the constructor.

## 1. Create one node pair per icdep of this kind

```
id = 1
for op in epoch.body:
    if op is not IcDepOp or op.kind != kind: skip
    graph.insert_node(Anchor(first_instr, first_or, first_mt, first_ir), id, First)
    graph.insert_node(Anchor(last_instr,  last_or,  last_mt,  last_ir ), id, Last)
    id += 1
```

## 2. For each non-sentinel node, DFS through cstrs to discover edges

```
for node in graph:
    if node is a sentinel: skip

    stack   = [ RangeRef(node.instr, node.mt,
                         lo=node.ir, hi=node.ir) ]
    visited = {}            # cycle guard

    while stack not empty:
        current = stack.pop()
        if current in visited: continue
        visited.add(current)

        # --- expand: follow every cstr whose src matches current ---
        for cstr in epoch.body where cstr is CstrOp:

            # If delay==[0,0] the cstr is bilateral; try src→dst then dst→src.
            for (src_ar, dst_ar) in directions(cstr):
                if src_ar.instr != current.instr:  continue
                if src_ar.mt != current.mt:  continue    # MT = event id

                # cstrs with no IR indices carry no range: forward dst as-is.
                if src_ar.ir is empty:
                    stack.push(RangeRef(dst_ar.instr, dst_ar.mt,
                                        lo=dst_ar.ir_lo, hi=dst_ar.ir_hi))
                    continue

                # Gate: drop only when src lies strictly in current's past
                # on every IR dim that propagates to dst.
                n = min(|src.ir_dims|, |dst.ir_dims|)
                if range_strictly_after(current.lo[:n], src.ir_hi[:n]):
                    continue

                # Map current → matching src element → corresponding dst coord.
                matched_src[i] = clamp(current.lo[i], src.ir_lo[i], src.ir_hi[i])
                new_lo         = map_index(matched_src, src.ir_lo, src.ir_hi,
                                                       dst.ir_lo, dst.ir_hi)
                stack.push(RangeRef(dst_ar.instr, dst_ar.mt,
                                    lo=new_lo, hi=dst_ar.ir_hi))

        # --- emit: connect node to any candidate whose anchor lies at
        #     or past current.lo in the same (instr, MT) space ---
        for candidate in graph:
            if candidate == node: continue
            if candidate.instr != current.instr:  continue
            if candidate.mt != current.mt:  continue
            if |candidate.ir| != |current.lo|: continue
            if all i: candidate.ir[i] >= current.lo[i]:
                graph.insert_edge(node.key, candidate.key)
```

## 3. Wire orphans to the sentinels

```
for n in graph:
    if n is a sentinel: skip
    if n has no incoming edges: graph.insert_edge(start, n)
    if n has no outgoing edges: graph.insert_edge(n, end)
```

## Helpers referenced above

| Name                    | Behavior                                                                       |
|-------------------------|--------------------------------------------------------------------------------|
| `directions(cstr)`      | yields `(src,dst)` and, if `cstr.delay == [0,0]`, also `(dst,src)`             |
| `range_strictly_after(a, b)` | true iff `a[i] > b[i]` on every dim                                       |
| `clamp(v, lo, hi)`      | `min(max(v, lo), hi)`                                                          |
| `map_index(idx, src_lo, src_hi, dst_lo, dst_hi)` | row-major flat-index correspondence between two ranges with equal element count |
