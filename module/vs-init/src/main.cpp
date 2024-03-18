// Copyright (C) 2021 Yu Yang
//
// This file is part of Vesyla.
//
// Vesyla is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// Vesyla is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with Vesyla.  If not, see <http://www.gnu.org/licenses/>.

#include "main.hpp"

using namespace std;
using namespace boost::program_options;
namespace fs = std::filesystem;

// Initialize logging system
INITIALIZE_EASYLOGGINGPP

int main(int argc, char **argv) {

  // Set Logger
  el::Configurations c;
  c.setToDefault();
  c.parseFromText(LOGGING_CONF);
  el::Loggers::reconfigureLogger("default", c);
  el::Loggers::addFlag(el::LoggingFlag::ColoredTerminalOutput);

  // Argument parsing
  bool force;
  string path_output;
  string style;
  options_description desc("Options");
  desc.add_options()("help,h", "Print help messages")(
      "force,f", bool_switch()->default_value(false), "Force Initialization")(
      "output,o", value<string>(&path_output)->default_value("."),
      "Output directory")("style,s", value<string>(&style)->required(),
                          "Program style, like vs-vesyla, vs-imecc, ...");
  variables_map vm;
  try {
    parsed_options parsed = parse_command_line(argc, argv, desc);
    store(parsed, vm);
    if (vm.count("help")) {
      cout << endl
           << "\e[1mVeInit (version: " << VERSION_MAJOR << "." << VERSION_MINOR
           << "." << VERSION_PATCH << ")\e[0m" << endl
           << endl
           << "================================================================"
              "================="
           << endl
           << "VeInit is distributed under \e[1m" << LICENSE_NAME
           << "\e[0m license agreement:" << endl
           << LICENSE_DESC << endl
           << "================================================================"
              "================="
           << endl
           << "Usage:" << endl
           << "  ./VeInit [options]" << endl
           << endl
           << desc << endl
           << "================================================================"
              "================="
           << endl
           << endl;
      return (0);
    }
    notify(vm);
  } catch (error &e) {
    LOG(ERROR) << endl;
    LOG(ERROR) << e.what() << endl;
    LOG(ERROR) << endl;
    LOG(ERROR) << desc << endl;
    return (-1);
  }

  force = vm["force"].as<bool>();

  // Real algorithm
  vesyla::util::SysPath sys_path;
  lock(force, path_output);
  initialize(style, path_output);
  return 0;
}

void lock(bool force, string path_output) {
  if (std::filesystem::exists(path_output + "/.lock")) {
    if (!force) {
      LOG(FATAL) << "Directory has already been initialized. If you want to "
                    "replace it, run the program with argument \"-f\".";
    }
  } else {
    fs::create_directory(path_output);
    time_t rawtime;
    struct tm *timeinfo;
    char buffer[80];
    time(&rawtime);
    timeinfo = localtime(&rawtime);
    strftime(buffer, sizeof(buffer), "%d-%m-%Y %H:%M:%S", timeinfo);
    std::string str(buffer);
    ofstream ofs(path_output + "/.lock");
    ofs << str;
    ofs.close();
  }
}

void initialize(string style, string path_output) {
  vesyla::util::GlobalVar glv;
  string path_template;

  glv.gets("$HOME_DIR", path_template);
  path_template = path_template + ".config/vesyla-suite/template/" + style;
  if (!fs::is_directory(path_template)) {
    glv.gets("$PROG_DIR", path_template);
    path_template = path_template + "share/vesyla-suite/template/" + style;
    if (!fs::is_directory(path_template)) {
      LOG(FATAL) << "Initialization template for \"" + style + "\" not found!";
    }
  }

  for (const auto &entry : fs::directory_iterator(path_template)) {
    fs::copy(entry.path(), fs::path(path_output) / entry.path().filename(),
             fs::copy_options::overwrite_existing |
                 fs::copy_options::recursive);
  }
}
