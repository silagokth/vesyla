#include "schedule/Scheduler.hpp"
#include "util/Common.hpp"
#include <string>

int main(int argc, char **argv) {

  // seed the random number generator
  auto seed = std::chrono::system_clock::now().time_since_epoch().count();
  srand(static_cast<unsigned int>(seed));

  // Set up logging system
  static plog::ColorConsoleAppender<plog::TxtFormatter> consoleAppender;
  plog::init(plog::debug).addAppender(&consoleAppender);

  // Parsing command line arguments
  vesyla::util::MiniArgs args;
  args.parse(argc, argv);

  if (args.flag("h") || args.flag("help")) {
    LOG_INFO << "Usage: vs-compile --arch FILE --isa FILE --pasm FILE "
                "[--output DIR]";
    LOG_INFO << "Or";
    LOG_INFO << "vs-compile --arch FILE --isa FILE --cpp FILE "
                "[--output DIR]";
    return 0;
  }

  std::string arch_file = args.get("arch", args.get("a"));
  std::string isa_file = args.get("isa", args.get("i"));
  std::string pasm_file = args.get("pasm", args.get("p"));
  std::string cpp_file = args.get("cpp", args.get("c"));
  std::string output_dir = args.get("output", args.get("o", "."));

  if (arch_file.empty() || isa_file.empty()) {
    LOG_FATAL << "Required arguments missing, see --help for usage.";
    return -1;
  }

  // File existence checks
  if (!std::filesystem::exists(arch_file)) {
    LOG_FATAL << "Error: Architecture file does not exist: ";
    return -1;
  }
  if (!std::filesystem::exists(isa_file)) {
    LOG_FATAL << "Error: ISA file does not exist: " << isa_file;
    return -1;
  }

  // Create output directory if it doesn't exist
  if (!std::filesystem::exists(output_dir)) {
    std::filesystem::create_directories(output_dir);
  }
  vesyla::util::GlobalVar::puts("__OUTPUT_DIR__", output_dir);

  if (!cpp_file.empty()) {
    LOG_FATAL << "Compilation from C++ model is not supported right now!";
    return -1;
  }

  if (!pasm_file.empty() && !std::filesystem::exists(pasm_file)) {
    if (!std::filesystem::exists(pasm_file)) {
      LOG_FATAL << "Error: PASM file does not exist: " << pasm_file;
      return -1;
    }
  }

  vesyla::pasm::Config cfg;
  cfg.set_arch_json(arch_file);
  cfg.set_isa_json(isa_file);

  vesyla::schedule::Scheduler scheduler;
  scheduler.run(pasm_file, output_dir);

  // clean up temporary directories
  std::string temp_dir = vesyla::util::SysPath::temp_dir();
  if (std::filesystem::exists(temp_dir)) {
    std::filesystem::remove_all(temp_dir);
  }

  return 0;
}
