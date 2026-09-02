#include "PatternLibrary.hpp"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "vesyla/Dialect/Drra/IR/DrraOps.hpp"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include "json/json.hpp"

#include <algorithm>
#include <fstream>
#include <map>

namespace vesyla {
namespace conversion {
namespace select_instructions {

namespace {

// Functions that are loaded but never matched, per resource kind. Each
// describes a datapath that a program written in affine + arith has no way to
// express, so registering them would only produce spurious matches.
struct SkipRule {
  llvm::StringRef kindPrefix;
  llvm::StringRef function;
  llvm::StringRef reason;
};

const SkipRule kSkipRules[] = {
    {"", "idle", "computes nothing: empty body, no root to match"},
    {"rf", "conf",
     "writes a register from the instruction stream; its address and value are "
     "match-derived, which the emission attributes cannot yet express"},
    {"iosram", "sram_read", "internal io<->sram staging, AGU-driven"},
    {"iosram", "sram_write", "internal io<->sram staging, AGU-driven"},
    {"iosram", "bulk_read",
     "fabric path; needs a buffer identity the program does not yet give"},
    {"iosram", "bulk_write",
     "fabric path; needs a buffer identity the program does not yet give"},
};

llvm::StringRef skipReason(llvm::StringRef kind, llvm::StringRef function) {
  for (const SkipRule &rule : kSkipRules) {
    if (rule.function != function)
      continue;
    if (rule.kindPrefix.empty() || kind.starts_with(rule.kindPrefix))
      return rule.reason;
  }
  return {};
}

// The io resources address an external buffer rather than local storage, so
// their patterns only apply where the program says it is touching one.
bool isIoKind(llvm::StringRef kind) {
  return kind == "io" || kind.starts_with("iosram");
}

// Resolve a segment value against the resource's own isa.json: the segment must
// exist on that instruction, and a value that names a verbo_map entry must
// resolve to a key that is actually declared.
mlir::LogicalResult validateSegments(const nlohmann::json &isa,
                                     llvm::StringRef instr,
                                     mlir::DictionaryAttr segments,
                                     llvm::StringRef where,
                                     llvm::raw_ostream &diag) {
  // Every lookup below is guarded: with -fno-exceptions nlohmann aborts on a
  // missing key rather than throwing, so a malformed isa.json would take the
  // compiler down instead of producing a diagnostic.
  if (!isa.contains("instructions") || !isa["instructions"].is_array())
    return diag << where << ": isa.json has no \"instructions\" array\n",
           mlir::failure();

  const nlohmann::json *found = nullptr;
  for (const nlohmann::json &candidate : isa["instructions"]) {
    if (candidate.value("name", "") == instr.str()) {
      found = &candidate;
      break;
    }
  }
  if (!found)
    return diag << where << ": isa.json declares no \"" << instr
                << "\" instruction\n",
           mlir::failure();

  if (!found->contains("segments") || !(*found)["segments"].is_array())
    return diag << where << ": \"" << instr << "\" has no \"segments\" array\n",
           mlir::failure();

  for (mlir::NamedAttribute segment : segments) {
    llvm::StringRef segName = segment.getName();
    const nlohmann::json *segJson = nullptr;
    for (const nlohmann::json &s : (*found)["segments"]) {
      if (s.value("name", "") == segName.str()) {
        segJson = &s;
        break;
      }
    }
    if (!segJson)
      return diag << where << ": \"" << instr << "\" has no segment \""
                  << segName << "\"\n",
             mlir::failure();

    auto value = llvm::dyn_cast<mlir::IntegerAttr>(segment.getValue());
    if (!value)
      return diag << where << ": segment \"" << segName
                  << "\" must be an integer\n",
             mlir::failure();

    // A segment with a verbo_map only accepts the keys it declares.
    if (!segJson->contains("verbo_map") || !(*segJson)["verbo_map"].is_array())
      continue;
    const int64_t key = value.getInt();
    bool declared = false;
    for (const nlohmann::json &entry : (*segJson)["verbo_map"])
      declared |= entry.value("key", -1) == key;
    if (!declared)
      return diag << where << ": " << key << " is not a declared value of \""
                  << segName << "\"\n",
             mlir::failure();
  }
  return mlir::success();
}

// Check the shape of a replacement rop's `uses` list: the parts of the resource
// the operation holds while it is active, which design-space exploration
// intersects to decide whether two operations can share an instance.
//
// The strings themselves are not checked against anything. There is nothing to
// check them against -- a resource's parts are its own business, and the
// compiler only ever asks whether two of them are spelled the same. What is
// worth catching is the shape, because a `uses` that is not an array of strings
// reads as no parts at all, and no parts means the operation is taken to hold
// the whole resource. That failure is silent and in the safe direction, so it
// would show up as a binding that needs more instances than it should rather
// than as anything obviously wrong.
mlir::LogicalResult validateUses(drra::RopOp rop, llvm::StringRef where,
                                 llvm::raw_ostream &diag) {
  mlir::Attribute attr = rop->getAttr("uses");
  if (!attr)
    return mlir::success();

  auto array = llvm::dyn_cast<mlir::ArrayAttr>(attr);
  if (!array)
    return diag << where
                << ": \"uses\" must be an array of strings naming the parts "
                   "of the resource the operation holds\n",
           mlir::failure();

  for (mlir::Attribute entry : array)
    if (!llvm::isa<mlir::StringAttr>(entry))
      return diag << where << ": \"uses\" must hold only strings\n",
             mlir::failure();

  return mlir::success();
}

// A rank-0 memref block argument is the resource's own register -- the dpu's
// accumulate register is the only one today. Returns the argument, its
// write-back, and whether the body also reads it.
//
// Reading is deliberately not required: @mac both reads and writes, but @rst
// only writes, and treating @rst's store as an ordinary body op would root the
// pattern on the store instead of on the constant it stores.
struct AccumulatorInfo {
  mlir::BlockArgument arg;
  mlir::Operation *store = nullptr;
  bool read = false;
};

AccumulatorInfo findAccumulator(mlir::func::FuncOp func) {
  for (mlir::BlockArgument arg : func.getArguments()) {
    auto memref = llvm::dyn_cast<mlir::MemRefType>(arg.getType());
    if (!memref || memref.getRank() != 0)
      continue;

    AccumulatorInfo info;
    info.arg = arg;
    for (mlir::Operation *user : arg.getUsers()) {
      if (llvm::isa<mlir::memref::LoadOp>(user))
        info.read = true;
      else if (auto s = llvm::dyn_cast<mlir::memref::StoreOp>(user))
        info.store = s;
    }
    return info;
  }
  return {};
}

} // namespace

bool Pattern::guardedByBufferRole() const { return isIoKind(kind); }

mlir::LogicalResult PatternLibrary::load(llvm::StringRef componentPath,
                                         mlir::MLIRContext *ctx,
                                         llvm::raw_ostream &diag) {
  llvm::SmallString<128> resources(componentPath);
  llvm::sys::path::append(resources, "resources");

  std::error_code ec;
  std::vector<std::pair<std::string, std::string>> found; // (mlir, isa)
  for (llvm::sys::fs::directory_iterator it(resources, ec), end;
       it != end && !ec; it.increment(ec)) {
    llvm::StringRef dir = it->path();
    llvm::StringRef kind = llvm::sys::path::filename(dir);

    llvm::SmallString<128> mlirPath(dir), isaPath(dir);
    llvm::sys::path::append(mlirPath, kind + ".mlir");
    llvm::sys::path::append(isaPath, "isa.json");
    if (llvm::sys::fs::exists(mlirPath) && llvm::sys::fs::exists(isaPath))
      found.push_back({std::string(mlirPath), std::string(isaPath)});
  }
  if (ec)
    return diag << "select-instructions: cannot read " << resources << ": "
                << ec.message() << "\n",
           mlir::failure();

  // Deterministic order, so a diagnostic reads the same on every run.
  std::sort(found.begin(), found.end());
  if (found.empty())
    return diag << "select-instructions: no resource patterns under "
                << resources
                << " -- is VESYLA_SUITE_PATH_COMPONENTS pointing at a built "
                   "component library?\n",
           mlir::failure();

  for (const auto &[mlirPath, isaPath] : found)
    if (mlir::failed(loadResource(mlirPath, isaPath, ctx, diag)))
      return mlir::failure();

  // Group into tiers by descending benefit.
  std::map<int64_t, std::vector<const Pattern *>, std::greater<int64_t>> byBenefit;
  for (const auto &p : patterns_)
    byBenefit[p->benefit].push_back(p.get());
  for (auto &[benefit, group] : byBenefit)
    tiers_.push_back(std::move(group));

  return mlir::success();
}

mlir::LogicalResult PatternLibrary::loadResource(llvm::StringRef mlirPath,
                                                 llvm::StringRef isaPath,
                                                 mlir::MLIRContext *ctx,
                                                 llvm::raw_ostream &diag) {
  auto module = mlir::parseSourceFile<mlir::ModuleOp>(mlirPath, ctx);
  if (!module)
    return diag << "select-instructions: failed to parse " << mlirPath << "\n",
           mlir::failure();

  nlohmann::json isa;
  {
    std::ifstream in(isaPath.str());
    if (!in)
      return diag << "select-instructions: cannot open " << isaPath << "\n",
             mlir::failure();
    // LLVM builds with -fno-exceptions, so parse in the non-throwing mode and
    // test the discarded sentinel instead of catching.
    isa = nlohmann::json::parse(in, /*cb=*/nullptr, /*allow_exceptions=*/false);
    if (isa.is_discarded())
      return diag << "select-instructions: " << isaPath
                  << ": not valid JSON\n",
             mlir::failure();
  }

  // The resource kind is the module's own symbol name -- it is not repeated as
  // an attribute anywhere.
  std::optional<llvm::StringRef> symName = module->getSymName();
  if (!symName)
    return diag << "select-instructions: " << mlirPath
                << ": the top-level module needs a name (module @<kind>)\n",
           mlir::failure();
  const std::string kind = symName->str();

  // A resource file is two nested modules: the shapes to look for, and what
  // each turns into.
  mlir::ModuleOp matchModule, replaceModule;
  for (mlir::ModuleOp nested : module->getOps<mlir::ModuleOp>()) {
    std::optional<llvm::StringRef> name = nested.getSymName();
    if (!name)
      continue;
    if (*name == "match")
      matchModule = nested;
    else if (*name == "replace")
      replaceModule = nested;
  }
  if (!matchModule || !replaceModule)
    return diag << "select-instructions: " << mlirPath
                << ": expected a @match and a @replace module inside @" << kind
                << "\n",
           mlir::failure();

  for (mlir::func::FuncOp matchFn : matchModule.getOps<mlir::func::FuncOp>()) {
    const std::string name = matchFn.getName().str();
    const std::string where = kind + ".mlir @" + name;

    auto benefit = matchFn->getAttrOfType<mlir::IntegerAttr>("benefit");
    if (!benefit)
      return diag << where << ": missing a benefit attribute\n",
             mlir::failure();

    // Every match needs a counterpart, whether or not it is selectable: the
    // pairing is checked before the skip list, so a resource cannot describe a
    // shape it has no instruction for.
    auto replaceFn = replaceModule.lookupSymbol<mlir::func::FuncOp>(name);
    if (!replaceFn)
      return diag << where << ": no function of that name in @replace\n",
             mlir::failure();
    if (replaceFn.getFunctionType().getInputs() !=
        matchFn.getFunctionType().getInputs())
      return diag << where
                  << ": @replace must take the same arguments as @match -- "
                     "argument i stands for whatever match argument i bound "
                     "to\n",
             mlir::failure();

    // Read the instruction segments off the replacement's rops and check them
    // against this resource's own isa.json.
    mlir::DictionaryAttr conf, evt;
    unsigned rops = 0;
    mlir::WalkResult walk =
        replaceFn.walk([&](drra::RopOp rop) -> mlir::WalkResult {
          ++rops;
          if (mlir::DictionaryAttr c = rop.getConfAttr()) {
            conf = c;
            if (mlir::failed(validateSegments(isa, "conf", c, where, diag)))
              return mlir::WalkResult::interrupt();
          }
          if (mlir::DictionaryAttr e = rop.getEvtAttr()) {
            evt = e;
            if (mlir::failed(validateSegments(isa, "evt", e, where, diag)))
              return mlir::WalkResult::interrupt();
          }
          if (mlir::failed(validateUses(rop, where, diag)))
            return mlir::WalkResult::interrupt();
          return mlir::WalkResult::advance();
        });
    if (walk.wasInterrupted())
      return mlir::failure();
    if (rops == 0)
      return diag << where << ": @replace builds no drra.rop\n",
             mlir::failure();
    if (!conf && !evt)
      return diag << where
                  << ": the replacement rop needs a conf or an evt -- without "
                     "one it lowers to no instruction\n",
             mlir::failure();

    // Paired and validated; now decide whether a program can actually reach it.
    if (llvm::StringRef reason = skipReason(kind, name); !reason.empty()) {
      skipped_.push_back({where, reason.str()});
      continue;
    }

    auto pattern = std::make_unique<Pattern>();
    pattern->kind = kind;
    pattern->name = name;
    pattern->benefit = benefit.getInt();
    pattern->conf = conf;
    pattern->evt = evt;
    pattern->match = matchFn;
    pattern->replace = replaceFn;
    AccumulatorInfo acc = findAccumulator(matchFn);
    pattern->accumulator = acc.arg;
    pattern->accumulatorStore = acc.store;
    pattern->readsAccumulator = acc.read;

    // The root is what a match is anchored at. A function that returns a value
    // roots on that value's producer; a void function roots on its last real
    // body op, skipping the accumulator write-back, which the accumulator rule
    // consumes rather than matches.
    mlir::Block &body = matchFn.getBody().front();
    mlir::Operation *terminator = body.getTerminator();
    if (terminator->getNumOperands() == 1) {
      pattern->root = terminator->getOperand(0).getDefiningOp();
    } else {
      for (mlir::Operation &op : llvm::reverse(body.without_terminator())) {
        if (&op == pattern->accumulatorStore)
          continue;
        pattern->root = &op;
        break;
      }
    }
    if (!pattern->root) {
      skipped_.push_back({where, "no root op to match"});
      continue;
    }

    patterns_.push_back(std::move(pattern));
  }

  // The other direction: a replacement nothing matches is dead weight, and
  // usually a renamed or misspelled match.
  for (mlir::func::FuncOp replaceFn :
       replaceModule.getOps<mlir::func::FuncOp>()) {
    if (!matchModule.lookupSymbol<mlir::func::FuncOp>(replaceFn.getName()))
      return diag << kind << ".mlir @" << replaceFn.getName()
                  << ": in @replace with no function of that name in @match\n",
             mlir::failure();
  }

  modules_.push_back(std::move(module));
  return mlir::success();
}

} // namespace select_instructions
} // namespace conversion
} // namespace vesyla
