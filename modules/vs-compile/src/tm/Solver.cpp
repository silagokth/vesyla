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

void turn_to_valid_json(string json_input_filename,
                        string json_output_filename) {
  // Get the input file stream
  std::ifstream input(json_input_filename);
  std::ofstream output(json_output_filename);

  // Read entire input into string
  std::stringstream buffer;
  buffer << input.rdbuf();
  std::string invalidJson = buffer.str();

  // Remove leading and trailing whitespace
  size_t start = invalidJson.find_first_not_of(" \t\n\r");
  size_t end = invalidJson.find_last_not_of(" \t\n\r");

  if (start == std::string::npos) {
    output << "[]";
    return;
  }

  std::string trimmed = invalidJson.substr(start, end - start + 1);

  // Simple approach: wrap in brackets and add commas between }{ patterns
  std::string result = "[" + trimmed + "]";

  // Find and replace "}  {" patterns with "}, {"
  size_t pos = 0;
  while ((pos = result.find("}", pos)) != std::string::npos) {
    size_t nextBrace = result.find("{", pos + 1);
    if (nextBrace != std::string::npos) {
      // Check if there's only whitespace between } and {
      bool onlyWhitespace = true;
      for (size_t i = pos + 1; i < nextBrace; ++i) {
        if (!std::isspace(result[i])) {
          onlyWhitespace = false;
          break;
        }
      }

      if (onlyWhitespace) {
        result.replace(pos, nextBrace - pos, "}, ");
        pos += 3; // Move past the ", "
      } else {
        pos++;
      }
    } else {
      break;
    }
  }

  output << result;

  input.close();
  output.close();
}

bool check_mzn_solution_valid(nlohmann::json &json_output, string &solution) {
  bool result = false;
  // Find all the "type" entries in the JSON output
  for (const auto &entry : json_output.items()) {
    for (auto sub_entry : entry.value().items()) {
      if (sub_entry.key() == "type") {
        if (sub_entry.value() == "solution") {
          result = true; // Found a solution
        }
      } else if (sub_entry.key() == "status") {
        if (sub_entry.value() == "UNSATISFIABLE") {
          solution = "UNSATISFIABLE";
          result = false;
          break; // No solution found
        }
      } else if (sub_entry.key() == "warning") {
        LOG_WARNING << sub_entry.value();
      } else if (sub_entry.key() == "output") {
        auto sub_sub_entry = sub_entry.value().find("dzn");
        solution = sub_sub_entry != sub_entry.value().end()
                       ? sub_sub_entry->get<string>()
                       : "no solution found";
      }
    }
  }
  return result; // No solution found
}

void write_minizinc_files(TimingModel &tm, string mzn_filename,
                          string dzn_filename, string json_filename,
                          bool allow_act_mode_2 = false) {
  // Create the MiniZinc model file
  std::ofstream mzn_file(mzn_filename);
  if (!mzn_file.is_open()) {
    LOG_FATAL << "Failed to create MiniZinc model file: " << mzn_filename;
    exit(-1);
  }

  // Create the MiniZinc data file
  std::ofstream dzn_file(dzn_filename);
  if (!dzn_file.is_open()) {
    LOG_FATAL << "Failed to create MiniZinc data file: " << dzn_filename;
    exit(-1);
  }

  // Create the JSON output file
  std::ofstream json_file(json_filename);
  if (!json_file.is_open()) {
    LOG_FATAL << "Failed to create JSON output file: " << json_filename;
    exit(-1);
  }

  // Write the MiniZinc model content
  tm.to_mzn(mzn_file, dzn_file, allow_act_mode_2);

  // Close all files
  mzn_file.close();
  dzn_file.close();
  json_file.close();
}

void run_minizinc(string mzn_filename, string dzn_filename,
                  string json_filename) {
  // run minizinc command and collect the json output
  LOG_DEBUG << "Running Minizinc command with input files: " << mzn_filename
            << " and " << dzn_filename;
  string command = "minizinc --json-stream --solver cp-sat " + mzn_filename +
                   " " + dzn_filename + " > " + json_filename;
  int result = system(command.c_str());

  LOG_DEBUG << "Minizinc command executed: " << command;
  LOG_DEBUG << "Minizinc command output file: " << json_filename;
  if (result == -1) {
    LOG_ERROR << "Failed to execute Minizinc command.";
    exit(-1);
  } else if (!WIFEXITED(result) || WEXITSTATUS(result) != 0) {
    LOG_ERROR << "Minizinc command failed with error code: "
              << WEXITSTATUS(result);
    exit(-1);
  }
}

nlohmann::json get_json_from_file(const string &filename) {
  // Read the JSON file and return the parsed JSON object
  std::ifstream json_file(filename);
  if (!json_file.is_open()) {
    LOG_FATAL << "Failed to open JSON file: " << filename;
    exit(-1);
  }
  nlohmann::json json_output;
  json_file >> json_output;
  json_file.close();
  return json_output;
}

unordered_map<string, string> Solver::solve(TimingModel &tm,
                                            bool allow_act_mode_2) {
  // create two temporary files with random names
  string random_id = get_random_string(10);
  string mzn_filename = tmp_path + random_id + ".mzn";
  string dzn_filename = tmp_path + random_id + ".dzn";
  string json_filename = tmp_path + random_id + ".json";
  string edited_json_filename = tmp_path + random_id + "_fixed.json";

  write_minizinc_files(tm, mzn_filename, dzn_filename, json_filename);
  run_minizinc(mzn_filename, dzn_filename, json_filename);
  turn_to_valid_json(json_filename, edited_json_filename);
  nlohmann::json json_output = get_json_from_file(edited_json_filename);

  string solution;
  if (!check_mzn_solution_valid(json_output, solution)) {
    solution.clear();
    json_output.clear();

    // rerun minizinc allowing act mode 2
    write_minizinc_files(tm, mzn_filename, dzn_filename, json_filename, true);
    run_minizinc(mzn_filename, dzn_filename, json_filename);
    turn_to_valid_json(json_filename, edited_json_filename);
    json_output = get_json_from_file(edited_json_filename);

    if (!check_mzn_solution_valid(json_output, solution)) {
      LOG_ERROR << "Minizinc command failed to find a solution.";
      LOG_ERROR << "Input MZN file: " << mzn_filename;
      LOG_ERROR << "Input DZN file: " << dzn_filename;
      LOG_ERROR << "Output: " << json_output.dump(4);
      LOG_FATAL << "Minizinc command failed.";
      exit(-1);
    }
  }

  LOG_DEBUG << "Minizinc command found a solution: " << solution;

  // get output
  string output_str = solution;

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
