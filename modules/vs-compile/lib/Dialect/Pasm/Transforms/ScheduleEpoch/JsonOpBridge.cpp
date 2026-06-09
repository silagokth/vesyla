#include "ScheduleEpochPassDetail.hpp"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/Support/raw_ostream.h"
#include <cctype>

namespace vesyla::pasm::schedule_epoch_detail {

nlohmann::json
ScheduleEpochPassRewriter::op2json(::mlir::Operation *op) const {
  nlohmann::json op_json;
  if (auto rop_op = llvm::dyn_cast<::vesyla::pasm::RopOp>(op)) {
    op_json["kind"] = "rop";
    op_json["id"] = rop_op.getSymName().str();
    op_json["row"] = rop_op.getRow();
    op_json["col"] = rop_op.getCol();
    op_json["slot"] = rop_op.getSlot();
    op_json["port"] = rop_op.getPort();

    // get its internal block
    ::mlir::Region &ropBodyRegion = rop_op.getBody();
    ::mlir::Block *ropEntryBlock;
    if (!ropBodyRegion.empty()) {
      ropEntryBlock = &ropBodyRegion.front();

      op_json["body"] = nlohmann::json::array();
      for (::mlir::Operation &rop_child_op : *ropEntryBlock) {
        if (auto instr_op =
                llvm::dyn_cast<::vesyla::pasm::InstrOp>(&rop_child_op)) {
          std::string id = instr_op.getId().str();
          std::string type = instr_op.getType().str();
          std::unordered_map<std::string, std::string> param_map;
          for (auto &param : instr_op.getParam()) {
            std::string param_name = param.getName().str();
            ::mlir::Attribute param_value = param.getValue();
            if (auto str_attr =
                    llvm::dyn_cast<::mlir::StringAttr>(param_value)) {
              param_map[param_name] = str_attr.getValue().str();
            } else if (auto int_attr = llvm::dyn_cast<::mlir::IntegerAttr>(
                           param_value)) {
              param_map[param_name] = std::to_string(int_attr.getInt());
            } else {
              llvm::outs() << "Unsupported parameter type in InstrOp: "
                           << param_value << "\n";
              std::exit(EXIT_FAILURE);
            }
          }

          nlohmann::json instr_json;
          instr_json["id"] = id;
          instr_json["kind"] = type;
          instr_json["params"] = nlohmann::json::array();
          for (auto &param : param_map) {
            nlohmann::json param_json;
            param_json["name"] = param.first;
            param_json["value"] = param.second;
            instr_json["params"].push_back(param_json);
          }
          op_json["body"].push_back(instr_json);
        } else if (auto yield_op = llvm::dyn_cast<::vesyla::pasm::YieldOp>(
                       &rop_child_op)) {
          // DO NOTHING
        } else {
          llvm::outs() << "Illegal operation type in RopOp: "
                       << rop_child_op.getName() << "\n";
          std::exit(EXIT_FAILURE);
        }
      }
    }
  } else if (auto cop_op = llvm::dyn_cast<::vesyla::pasm::CopOp>(op)) {
    op_json["kind"] = "cop";
    op_json["id"] = cop_op.getId().str();
    op_json["row"] = cop_op.getRow();
    op_json["col"] = cop_op.getCol();

    // get its internal block
    ::mlir::Region &copBodyRegion = cop_op.getBody();
    ::mlir::Block *copEntryBlock;
    if (!copBodyRegion.empty()) {

      copEntryBlock = &copBodyRegion.front();

      op_json["body"] = nlohmann::json::array();
      for (::mlir::Operation &cop_child_op : *copEntryBlock) {
        if (auto instr_op =
                llvm::dyn_cast<::vesyla::pasm::InstrOp>(&cop_child_op)) {
          std::string id = instr_op.getId().str();
          std::string type = instr_op.getType().str();
          std::unordered_map<std::string, std::string> param_map;
          for (auto &param : instr_op.getParam()) {
            std::string param_name = param.getName().str();
            ::mlir::Attribute param_value = param.getValue();
            if (auto str_attr =
                    llvm::dyn_cast<::mlir::StringAttr>(param_value)) {
              param_map[param_name] = str_attr.getValue().str();
            } else if (auto int_attr = llvm::dyn_cast<::mlir::IntegerAttr>(
                           param_value)) {
              param_map[param_name] = std::to_string(int_attr.getInt());
            } else {
              llvm::outs() << "Unsupported parameter type in InstrOp: "
                           << param_value << "\n";
              std::exit(EXIT_FAILURE);
            }
          }

          nlohmann::json instr_json;
          instr_json["kind"] = type;
          instr_json["params"] = nlohmann::json::array();
          for (auto &param : param_map) {
            nlohmann::json param_json;
            param_json["name"] = param.first;
            param_json["value"] = param.second;
            instr_json["params"].push_back(param_json);
          }
          op_json["body"].push_back(instr_json);
        } else if (auto yield_op = llvm::dyn_cast<::vesyla::pasm::YieldOp>(
                       &cop_child_op)) {
          // DO NOTHING
        } else {
          llvm::outs() << "Illegal operation type in CopOp: "
                       << cop_child_op.getName() << "\n";
          std::exit(EXIT_FAILURE);
        }
      }
    }
  } else {
    llvm::outs() << "Unsupported operation type for JSON conversion: "
                 << op->getName() << "\n";
    std::exit(EXIT_FAILURE);
  }

  return op_json;
}

void ScheduleEpochPassRewriter::json2op(
    nlohmann::json op_json, ::mlir::PatternRewriter &rewriter) const {
  if (op_json["kind"].get<std::string>() == "rop") {
    auto rop_op = rewriter.create<::vesyla::pasm::RopOp>(
        rewriter.getUnknownLoc(),
        rewriter.getStringAttr(op_json["id"].get<std::string>()),
        rewriter.getI32IntegerAttr(op_json["row"].get<int>()),
        rewriter.getI32IntegerAttr(op_json["col"].get<int>()),
        rewriter.getI32IntegerAttr(op_json["slot"].get<int>()),
        rewriter.getI32IntegerAttr(op_json["port"].get<int>()),
        /*map=*/mlir::AffineMapAttr());

    // get its internal block
    ::mlir::Region &ropBodyRegion = rop_op.getBody();
    ::mlir::Block *ropEntryBlock;
    if (ropBodyRegion.empty()) {
      ropEntryBlock = rewriter.createBlock(&ropBodyRegion);
    } else {
      ropEntryBlock = &ropBodyRegion.front();
    }
    rewriter.setInsertionPointToEnd(ropEntryBlock);

    for (auto &instr : op_json["body"]) {
      std::unordered_map<std::string, std::string> param_map;
      for (auto &param : instr["params"]) {
        param_map[param["name"].get<std::string>()] =
            param["value"].get<std::string>();
      }

      ::mlir::StringAttr id = rewriter.getStringAttr(
          ::vesyla::util::Common::gen_random_string(8));
      ::mlir::StringAttr type =
          rewriter.getStringAttr(instr["kind"].get<std::string>());
      llvm::SmallVector<::mlir::NamedAttribute> attrs;

      for (const auto &param : param_map) {
        // check if it's a number or a string
        if (std::isdigit(param.second[0])) {
          attrs.push_back(rewriter.getNamedAttr(
              param.first,
              rewriter.getI32IntegerAttr(std::stoi(param.second))));
        } else {
          attrs.push_back(rewriter.getNamedAttr(
              param.first, rewriter.getStringAttr(param.second)));
        }
      }
      ::mlir::DictionaryAttr param = rewriter.getDictionaryAttr(attrs);
      rewriter.create<::vesyla::pasm::InstrOp>(
          rop_op.getLoc(),
          rewriter.getStringAttr(instr["id"].get<std::string>()),
          rewriter.getStringAttr(instr["kind"].get<std::string>()), param);
    }
    // insert a yield operation at the end of the RopOp
    rewriter.create<::vesyla::pasm::YieldOp>(rop_op.getLoc());

  } else if (op_json["kind"] == "cop") {
    auto cop_op = rewriter.create<::vesyla::pasm::CopOp>(
        rewriter.getUnknownLoc(),
        rewriter.getStringAttr(op_json["id"].get<std::string>()),
        rewriter.getI32IntegerAttr(op_json["row"].get<int>()),
        rewriter.getI32IntegerAttr(op_json["col"].get<int>()));
    // get its internal block
    ::mlir::Region &copBodyRegion = cop_op.getBody();
    ::mlir::Block *copEntryBlock;
    if (copBodyRegion.empty()) {
      copEntryBlock = rewriter.createBlock(&copBodyRegion);
    } else {
      copEntryBlock = &copBodyRegion.front();
    }
    rewriter.setInsertionPointToEnd(copEntryBlock);
    for (auto &instr : op_json["body"]) {
      std::unordered_map<std::string, std::string> param_map;
      for (auto &param : instr["params"]) {
        param_map[param["name"].get<std::string>()] =
            param["value"].get<std::string>();
      }

      ::mlir::StringAttr id = rewriter.getStringAttr(
          ::vesyla::util::Common::gen_random_string(8));
      ::mlir::StringAttr type =
          rewriter.getStringAttr(instr["kind"].get<std::string>());
      llvm::SmallVector<::mlir::NamedAttribute> attrs;

      for (const auto &param : param_map) {
        // check if it's a number or a string
        if (std::isdigit(param.second[0])) {
          attrs.push_back(rewriter.getNamedAttr(
              param.first,
              rewriter.getI32IntegerAttr(std::stoi(param.second))));
        } else {
          attrs.push_back(rewriter.getNamedAttr(
              param.first, rewriter.getStringAttr(param.second)));
        }
      }
      ::mlir::DictionaryAttr param = rewriter.getDictionaryAttr(attrs);
      rewriter.create<::vesyla::pasm::InstrOp>(
          cop_op.getLoc(),
          rewriter.getStringAttr(instr["id"].get<std::string>()),
          rewriter.getStringAttr(instr["kind"].get<std::string>()), param);
    }
    // insert a yield operation at the end of the CopOp
    rewriter.create<::vesyla::pasm::YieldOp>(cop_op.getLoc());
  } else {
    llvm::outs() << "Unsupported operation kind: "
                 << op_json["kind"].get<std::string>() << "\n";
    std::exit(EXIT_FAILURE);
  }
}

} // namespace vesyla::pasm::schedule_epoch_detail
