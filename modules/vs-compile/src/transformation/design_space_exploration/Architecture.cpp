#include "Architecture.hpp"

#include "vesyla/Support/Config.hpp"

namespace vesyla {
namespace transformation {
namespace dse {

namespace {

// Reading a field out of the architecture file, one type at a time.
//
// Every read checks the type before taking the value rather than letting
// nlohmann convert and throw: this library is built with -fno-exceptions, so a
// throw here would abort the compiler instead of reporting which part of the
// file is wrong. A field that is missing, null, or of the wrong type falls back
// to the default, and the caller turns an impossible default (a negative slot,
// an empty kind) into a diagnostic.
int int_or(const nlohmann::json &object, const char *key, int fallback) {
  if (!object.is_object() || !object.contains(key)) {
    return fallback;
  }
  const nlohmann::json &value = object.at(key);
  if (!value.is_number_integer()) {
    return fallback;
  }
  return value.get<int>();
}

bool bool_or(const nlohmann::json &object, const char *key, bool fallback) {
  if (!object.is_object() || !object.contains(key)) {
    return fallback;
  }
  const nlohmann::json &value = object.at(key);
  if (!value.is_boolean()) {
    return fallback;
  }
  return value.get<bool>();
}

std::string string_or(const nlohmann::json &object, const char *key,
                      const std::string &fallback) {
  if (!object.is_object() || !object.contains(key)) {
    return fallback;
  }
  const nlohmann::json &value = object.at(key);
  if (!value.is_string()) {
    return fallback;
  }
  return value.get<std::string>();
}

} // namespace

mlir::FailureOr<Architecture>
Architecture::from_config(mlir::Operation *diag_op) {
  ::vesyla::pasm::Config cfg;
  nlohmann::json arch_json = cfg.get_arch_json();

  if (!arch_json.is_object() || !arch_json.contains("cells")) {
    diag_op->emitError("design-space-exploration: the architecture file has no "
                       "\"cells\" section -- it must be the elaborated "
                       "arch.json (work/system/arch/arch.json), not the source "
                       "form");
    return mlir::failure();
  }

  Architecture arch;

  const nlohmann::json &parameters = arch_json.contains("parameters")
                                         ? arch_json.at("parameters")
                                         : nlohmann::json::object();
  arch.rows_ = int_or(parameters, "ROWS", 0);
  arch.cols_ = int_or(parameters, "COLS", 0);

  for (const auto &entry : arch_json.at("cells").items()) {
    const nlohmann::json &cell_entry = entry.value();
    if (!cell_entry.is_object() || !cell_entry.contains("coordinates") ||
        !cell_entry.contains("cell")) {
      diag_op->emitError("design-space-exploration: cell entry \"" +
                         entry.key() +
                         "\" has no coordinates or no cell body -- the "
                         "architecture file is not elaborated");
      return mlir::failure();
    }

    const nlohmann::json &coordinates = cell_entry.at("coordinates");
    int row = int_or(coordinates, "row", -1);
    int col = int_or(coordinates, "col", -1);
    if (row < 0 || col < 0) {
      diag_op->emitError("design-space-exploration: cell entry \"" +
                         entry.key() + "\" has no (row, col) coordinates");
      return mlir::failure();
    }

    const nlohmann::json &cell = cell_entry.at("cell");
    if (!cell.is_object() || !cell.contains("resources_list")) {
      // A cell with no resources is legal -- it just contributes nothing to
      // the instance table.
      continue;
    }

    for (const auto &resource_entry : cell.at("resources_list").items()) {
      const nlohmann::json &resource = resource_entry.value();
      ResourceInstance instance;
      instance.kind = string_or(resource, "kind", "");
      instance.row = row;
      instance.col = col;
      instance.slot = int_or(resource, "slot", -1);
      instance.size = int_or(resource, "size", 1);
      instance.io_input = bool_or(resource, "io_input", false);
      instance.io_output = bool_or(resource, "io_output", false);
      if (resource.is_object() && resource.contains("parameters")) {
        instance.parameters = resource.at("parameters");
      }

      if (instance.kind.empty() || instance.slot < 0) {
        diag_op->emitError("design-space-exploration: resource at cell (" +
                           std::to_string(row) + ", " + std::to_string(col) +
                           ") has no kind or no slot");
        return mlir::failure();
      }
      arch.instances_.push_back(std::move(instance));
    }
  }

  if (arch.instances_.empty()) {
    diag_op->emitError("design-space-exploration: the architecture file "
                       "declares no resources");
    return mlir::failure();
  }

  return arch;
}

llvm::SmallVector<unsigned>
Architecture::instances_of_kind(llvm::StringRef kind) const {
  llvm::SmallVector<unsigned> indices;
  for (unsigned i = 0; i < instances_.size(); ++i) {
    if (instances_[i].kind == kind) {
      indices.push_back(i);
    }
  }
  return indices;
}

} // namespace dse
} // namespace transformation
} // namespace vesyla
