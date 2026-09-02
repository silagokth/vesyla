#ifndef VESYLA_TRANSFORMATION_DESIGN_SPACE_EXPLORATION_PASSES
#define VESYLA_TRANSFORMATION_DESIGN_SPACE_EXPLORATION_PASSES

include "mlir/Pass/PassBase.td"

def DesignSpaceExplorationPass : Pass<"design-space-exploration-pass", "::mlir::ModuleOp"> {
  let summary = "Assign every selected instruction to a hardware resource.";
  let description = [{
    Runs on the module instruction selection produced, where every `drra.rop`
    says *what* it does -- its `kind`, and the `conf`/`evt` segments of its
    instruction -- but nothing says *where* it runs. This pass answers that, and
    the module it leaves behind carries a `resource` on every operation.

    Which resources exist is not decided here: allocation is read from the
    architecture file passed to the executable, since that file already fixes
    which resources the design has and where they sit. What is decided is which
    instance each operation runs on, and it is decided in two steps.

    First a *conflict graph* is built over each scope, and an edge in it says
    two nodes cannot be put on the same instance.

    A node is not an operation -- it is what has to be bound as a unit. Every
    access to one register file must land on the same register file, so the
    operations sharing a `storage` are merged into one node before the graph is
    built: a conflict graph can say two things must differ but never that two
    things must agree, so what must agree is merged rather than constrained.

    Two nodes conflict when they are of the same `kind` -- two of different
    kinds cannot share an instance whatever the program does -- and then either
    of the following:

      * they hold different storage. One instance holds one storage, so these
        never share, whether or not they ever run at the same time. Two arrays
        could in principle sit in one register file if their lifetimes were
        disjoint or their addresses could not collide, but neither is
        established here, so the conservative reading stands.
      * their occupancies overlap *and* they want the same part of the
        resource. Occupancy runs from where an operation runs until the last
        reader of its results has taken them, so two nodes whose stretches do
        not meet are never both running. Parts are what of the resource a node
        holds: a dpu has config registers, narrow inputs and outputs, and an AGU
        per event port, an rf has four access paths each with its own AGU and
        data wires, and a mac and a rst want none of the same ones. Each
        operation declares its parts in `uses`, written by the resource's own
        description in the component library; one that declares none is read as
        holding the whole resource.

    The graph is written out as Graphviz DOT under the `conflict_graph_dir`
    output path, one file per scope. Each node is labelled with what it is, the
    operations it was merged from, and the parts it holds; each edge with what
    the two contend for. So the picture says why every conflict is there and not
    only that it is.

    Binding is then a colouring of that graph: the instances of a kind are the
    colours, and two joined nodes must be given different ones. The colouring is
    greedy, taking the most contended nodes first, which is not optimal but
    never puts two conflicting operations together. When a kind runs out of
    instances the pass says which one ran short, rather than overcommitting an
    instance and letting it surface later as wrong results.

    What the graph does not know is timing. Its edges come from data flow, which
    orders operations only as far as their dependencies do, so two independent
    operations it leaves unjoined may still overlap once ScheduleEpochPass
    solves exact timing against each resource's timing model further down. It is
    a lower bound on the conflicts a binding must respect -- exact for the
    sequential reading of the program, and no stronger.

    Each `pasm.epoch` is handled on its own, since an epoch is the region
    downstream scheduling treats as a unit; a program with no epochs is handled
    as a whole.

    Everything this pass decides is annotated onto the operations, so a run is
    read back from the module it leaves behind.

    The pass ends by checking that every `drra.rop` came out bound. That check
    is the contract with everything downstream -- GenerateIcdepPass and
    DrraToPasmPass both read `resource` and cannot proceed without it -- so a
    gap is reported here, against the operation that has it, rather than
    surfacing as a confusing failure several passes later.
  }];

  let dependentDialects = ["::vesyla::drra::DrraDialect",
                           "::vesyla::pasm::PasmDialect"];
}

#endif // VESYLA_TRANSFORMATION_DESIGN_SPACE_EXPLORATION_PASSES
