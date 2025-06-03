#include "pasm/Dialect.hpp"
#include "pasm/Passes.hpp"
#include "schedule/Scheduler.hpp"
#include "util/Common.hpp"

int main(int argc, char **argv) {

  static plog::ColorConsoleAppender<plog::TxtFormatter> consoleAppender;
  plog::init(plog::debug).addAppender(&consoleAppender);

  vesyla::util::MiniArgs args;
  args.parse(argc, argv);

  if (args.flag("h") || args.flag("help")) {
    LOG_INFO << "Usage: vs-compile --arch FILE --isa FILE --pasm FILE "
                "[--output DIR]";
    return 0;
  }

  std::string arch_file = args.get("arch", args.get("a"));
  std::string isa_file = args.get("isa", args.get("i"));
  std::string pasm_file = args.get("pasm", args.get("p"));
  std::string output_dir = args.get("output", args.get("o", "."));

  if (arch_file.empty() || isa_file.empty() || pasm_file.empty()) {
    LOG_FATAL << "Required arguments missing";
    return 1;
  }

  // Print parsed arguments for debugging
  LOG_INFO << "Parsed arguments:";
  LOG_INFO << "  Architecture file: " << arch_file;
  LOG_INFO << "  ISA file: " << isa_file;
  LOG_INFO << "  PASM file: " << pasm_file;
  LOG_INFO << "  Output directory: " << output_dir;

  // File existence checks
  if (!std::filesystem::exists(arch_file)) {
    LOG_FATAL << "Error: Architecture file does not exist: ";
    return -1;
  }
  if (!std::filesystem::exists(isa_file)) {
    LOG_FATAL << "Error: ISA file does not exist: " << isa_file;
    return -1;
  }
  if (!std::filesystem::exists(pasm_file)) {
    LOG_FATAL << "Error: PASM file does not exist: " << pasm_file;
    return -1;
  }

  // Create output directory if it doesn't exist
  if (!std::filesystem::exists(output_dir)) {
    std::filesystem::create_directories(output_dir);
  }

  vesyla::pasm::Config cfg;
  cfg.set_arch_json(arch_file);
  cfg.set_isa_json(isa_file);

  // prepare the temporary directory
  std::string tmp_path = "/tmp/vesyla";
  if (!std::filesystem::exists(tmp_path)) {
    std::filesystem::create_directory(tmp_path);
  }

  vesyla::schedule::Scheduler scheduler;
  scheduler.run(pasm_file, output_dir);

  return 0;
}
