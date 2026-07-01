#include "ScheduleEpochPassDetail.hpp"
#include "vesyla/Analysis/TimingModel/TimingModel.hpp"
#include "vesyla/Support/Config.hpp"
#include "vesyla/Support/SysPath.hpp"
#include "llvm/Support/raw_ostream.h"

#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace vesyla::pasm::schedule_epoch_detail {

namespace {

// Extract the integers from a string, e.g. "[0: 95, 1: 91]" -> {0,95,1,91}.
std::vector<int> parse_int_array(const std::string &s) {
  std::vector<int> out;
  std::string num;
  for (char c : s) {
    if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) {
      num.push_back(c);
    } else if (!num.empty()) {
      out.push_back(std::stoi(num));
      num.clear();
    }
  }
  if (!num.empty()) {
    out.push_back(std::stoi(num));
  }
  return out;
}

// Parse a MiniZinc array string into an index -> value map. MiniZinc prints
// arrays with explicit element indices, e.g. "[0: 95, 1: 91, 2: 93]", so the
// integers alternate index, value, index, value, ... Falls back to positional
// indexing if no ':' is present.
std::unordered_map<int, int> parse_indexed_array(const std::string &s) {
  std::unordered_map<int, int> out;
  std::vector<int> nums = parse_int_array(s);
  if (s.find(':') != std::string::npos) {
    for (size_t i = 0; i + 1 < nums.size(); i += 2) {
      out[nums[i]] = nums[i + 1];
    }
  } else {
    for (size_t i = 0; i < nums.size(); i++) {
      out[static_cast<int>(i)] = nums[i];
    }
  }
  return out;
}

} // namespace

// Write the resolved schedule of one epoch to a JSON file for visualization.
// Layout: { epoch, total_latency,
//   resources:   [ { row, col, slot, operations: [ { name, port, start, end,
//                    duration } ] } ],
//   controllers: [ { row, col, instructions: [ { type, cycle, end,
//                    slot?, port? } ] } ] }.
// resources hold the scheduled RopOp datapath operations; controllers hold the
// per-cell COP control instructions (cycles resolved from schedule_table via
// the anchors in cop_instrs).
//
// Starts come from schedule_table (by op name); ends come from op_end_vec in
// the raw solver output `result` (indexed by op2idx). min_shift_time is the
// uniform per-epoch shift applied by synchronize so the earliest instruction
// lands at cycle 0; adding it makes start/end/latency match the absolute cycles
// in the emitted code. Durations are shift-invariant.
void dump_schedule_table(
    const std::string &tmp_path, const std::string &epoch_id,
    const std::unordered_map<std::string, int> &schedule_table,
    const std::unordered_map<std::string, std::string> &result,
    const ::vesyla::tm::TimingModel &model, int min_shift_time,
    const std::vector<CtrlInstr> &controller_instrs) {

  // The compile output directory is the parent of the "debug" directory.
  // tmp_path is the minizinc debug dir, e.g. "<output_dir>/debug/minizinc/".
  std::string trimmed = tmp_path;
  while (!trimmed.empty() && trimmed.back() == '/') {
    trimmed.pop_back();
  }
  std::filesystem::path output_dir =
      std::filesystem::path(trimmed).parent_path().parent_path();

  // The timetable location comes from the "output" section of config.json (with
  // "${key}" references resolved), e.g. "${compile_dir}/timetable" ->
  // "compile/timetable", relative to the compile output directory.
  ::vesyla::pasm::Config cfg;
  std::string timetable_dir = cfg.output_path("timetable_dir");

  std::filesystem::path timetable_path = output_dir / timetable_dir;
  std::error_code ec;
  std::filesystem::create_directories(timetable_path, ec);
  if (ec) {
    llvm::outs() << "Warning: could not create timetable directory "
                 << timetable_path.string() << ": " << ec.message() << "\n";
    return;
  }

  // Map operation name -> its index in the timing model (op2idx), matching the
  // order to_mzn used so the op_end_vec indices line up by name.
  std::unordered_map<std::string, int> op2idx;
  int idx = 0;
  for (auto it = model.operations.begin(); it != model.operations.end(); ++it) {
    op2idx[it->second.name] = idx;
    idx++;
  }

  // Resolved end times come from op_end_vec in the raw solver output, indexed by
  // op2idx. Note: op_end_vec == start + duration + 1 (the +1 is the act-issue
  // cycle).
  std::unordered_map<int, int> end_by_index;
  auto end_it = result.find("op_end_vec");
  if (end_it != result.end()) {
    end_by_index = parse_indexed_array(end_it->second);
  }

  // Group operations by the (row, col, slot) resource tuple.
  std::map<std::tuple<int, int, int>, nlohmann::json> resource_ops;
  for (auto it = schedule_table.begin(); it != schedule_table.end(); ++it) {
    const std::string &symbol = it->first;

    // Only operation start-time symbols carry a placement; skip anchors,
    // total_latency, inserted controller ids, and other intermediate variables.
    auto op_it = model.operations.find(symbol);
    if (op_it == model.operations.end()) {
      continue;
    }
    const ::vesyla::tm::Operation &o = op_it->second;
    int start_time = it->second + min_shift_time;

    nlohmann::json op_json;
    op_json["name"] = o.name;
    op_json["port"] = o.port;
    op_json["start"] = start_time;
    auto ix = op2idx.find(o.name);
    if (ix != op2idx.end()) {
      auto e = end_by_index.find(ix->second);
      if (e != end_by_index.end()) {
        int end_time = e->second + min_shift_time;
        op_json["end"] = end_time;
        op_json["duration"] = end_time - start_time;
      }
    }

    resource_ops[std::make_tuple(o.row, o.col, o.slot)].push_back(
        std::move(op_json));
  }

  // Assemble the document.
  nlohmann::json resources = nlohmann::json::array();
  for (auto it = resource_ops.begin(); it != resource_ops.end(); ++it) {
    nlohmann::json r;
    r["row"] = std::get<0>(it->first);
    r["col"] = std::get<1>(it->first);
    r["slot"] = std::get<2>(it->first);
    r["operations"] = std::move(it->second);
    resources.push_back(std::move(r));
  }

  // Controller instructions: the per-cell streams recovered from the
  // post-synchronize pasm.raw blocks. Cycles are already absolute, so they are
  // emitted verbatim. Grouped per cell so the visualizer can show one controller
  // lane per (row, col).
  std::map<std::pair<int, int>, nlohmann::json> controller_ops;
  for (auto it = controller_instrs.begin(); it != controller_instrs.end();
       ++it) {
    nlohmann::json instr_json;
    instr_json["type"] = it->type;
    instr_json["cycle"] = it->cycle;
    instr_json["end"] = it->end;
    // Target resource, present only for slot-targeting instructions
    // (evt/conf/rep/trans, ...); lets the visualizer echo it on that slot.
    if (it->slot >= 0) {
      instr_json["slot"] = it->slot;
      instr_json["port"] = it->port;
    }
    controller_ops[std::make_pair(it->row, it->col)].push_back(
        std::move(instr_json));
  }

  nlohmann::json controllers = nlohmann::json::array();
  for (auto it = controller_ops.begin(); it != controller_ops.end(); ++it) {
    nlohmann::json c;
    c["row"] = it->first.first;
    c["col"] = it->first.second;
    c["instructions"] = std::move(it->second);
    controllers.push_back(std::move(c));
  }

  nlohmann::json doc;
  doc["epoch"] = epoch_id;
  auto tl = schedule_table.find("total_latency");
  doc["total_latency"] =
      (tl != schedule_table.end()) ? tl->second + min_shift_time : -1;
  doc["resources"] = std::move(resources);
  doc["controllers"] = std::move(controllers);

  std::filesystem::path out_filename =
      timetable_path /
      (cfg.output_path("schedule_dump_prefix") + epoch_id + ".json");
  std::ofstream out(out_filename);
  if (!out.is_open()) {
    llvm::outs() << "Warning: could not write timetable dump to "
                 << out_filename.string() << "\n";
    return;
  }
  out << doc.dump(2);
  out.close();
  llvm::outs() << "Schedule timetable written to: " << out_filename.string()
               << "\n";

  // Best-effort render of the JSON just written to SVG/PNG. A missing script or
  // a failing render only logs a warning; it never aborts the compile.
  std::string timetable_script =
      ::vesyla::util::SysPath::prog_dir() + cfg.script_path("timetable");
  if (!std::filesystem::exists(timetable_script)) {
    llvm::outs() << "Warning: timetable visualization script not found: "
                 << timetable_script << "\n";
    return;
  }
  std::string cmd = "python3 " + timetable_script + " " + out_filename.string();
  int rc = std::system(cmd.c_str());
  if (rc != 0) {
    llvm::outs() << "Warning: timetable visualization failed (exit " << rc
                 << "): " << cmd << "\n";
  }
}

} // namespace vesyla::pasm::schedule_epoch_detail
