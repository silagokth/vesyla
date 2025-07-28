#include "Solver.hpp"

namespace vesyla {
namespace tm {

using namespace std;

string get_random_string(size_t length) {
  const string characters =
      "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
  string result;
  result.reserve(length);
  for (size_t i = 0; i < length; ++i) {
    result += characters[rand() % characters.size()];
  }
  return result;
}

unordered_map<string, string> Solver::solve(TimingModel &tm) {
  // create two temporary files with random names
  string random_id = get_random_string(10);
  string mzn_filename = tmp_path + "/" + random_id + ".mzn";
  string dzn_filename = tmp_path + "/" + random_id + ".dzn";
  string json_filename = tmp_path + "/" + random_id + ".json";

  std::ofstream mzn_file(mzn_filename);
  std::ofstream dzn_file(dzn_filename);

  if (!mzn_file.is_open() || !dzn_file.is_open()) {
    LOG_FATAL << "Failed to create temporary files: " << mzn_filename << " and "
              << dzn_filename << "\n";
    exit(-1);
  }

  if (tm.to_mzn(mzn_file, dzn_file) != 0) {
    LOG_FATAL << "Failed to convert TimingModel to MiniZinc.";
    exit(-1);
  }

  mzn_file.close();
  dzn_file.close();

  // run minizinc command and collect the json output
  LOG_DEBUG << "Running Minizinc command with input files: " << mzn_filename
            << " and " << dzn_filename;
  string command = "minizinc --json-stream --solver cp-sat " + mzn_filename +
                   " " + dzn_filename + " > " + json_filename;
  int result = system(command.c_str());

  LOG_DEBUG << "Minizinc command executed: " << command;
  LOG_DEBUG << "Minizinc command output file: " << json_filename;
  if (result != 0) {
    LOG_ERROR << "Minizinc command failed with error code: " << result;
  }

  // read the json output
  ifstream json_file(json_filename);
  if (!json_file.is_open()) {
    LOG_FATAL << "Failed to open json file.";
    exit(-1);
  }

  nlohmann::json json_output;
  json_file >> json_output;
  json_file.close();

  // check if the output type is solution
  if (json_output.find("type") == json_output.end() ||
      json_output["type"] != "solution") {
    LOG_ERROR << "Minizinc output is not a solution.";
    LOG_ERROR << "Input MZN file: " << mzn_filename;
    LOG_ERROR << "Input DZN file: " << dzn_filename;
    LOG_ERROR << "Output: " << json_output.dump(4);
    LOG_FATAL << "Minizinc command failed.";
    exit(-1);
  }

  // delete the temporary files
  // remove(input_filename.c_str());
  // remove(output_filename.c_str());

  // get output
  string output_str = json_output["output"]["dzn"].get<string>();

  // divide the output by delimiter \n
  vector<string> output_lines;
  string delimiter = "\n";
  size_t pos = 0;
  while ((pos = output_str.find(delimiter)) != string::npos) {
    output_lines.push_back(output_str.substr(0, pos));
    output_str.erase(0, pos + delimiter.length());
  }
  output_lines.push_back(output_str);

  // parse the output and store it in a unordered_map
  unordered_map<string, string> output_map;
  for (const auto &line : output_lines) {
    size_t pos = line.find("=");
    if (pos != string::npos) {
      string key = line.substr(0, pos);
      string value = line.substr(pos + 1);
      // remove leading and trailing whitespace
      key.erase(0, key.find_first_not_of(" \t"));
      key.erase(key.find_last_not_of(" \t") + 1);
      value.erase(0, value.find_first_not_of(" \t"));
      value.erase(value.find_last_not_of(" \t") + 1);
      // remove the semicolon at the end of the value
      if (value.back() == ';') {
        value.pop_back();
      }
      output_map[key] = value;
    }
  }

  return output_map;
}

} // namespace tm
} // namespace vesyla
