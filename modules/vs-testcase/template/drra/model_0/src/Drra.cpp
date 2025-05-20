#include "Drra.hpp"

void store_data(string file_, IO buffer) { io_to_file(buffer, file_); }
void load_data(string file_, IO &buffer) { buffer = file_to_io(file_); }

void reset(IO &buffer) { buffer.reset(); }

bool verify(string file0_, string file1_) {
  IO io0 = file_to_io(file0_);
  IO io1 = file_to_io(file1_);

  vector<size_t> active_rows_0 = io0.get_active_rows();
  vector<size_t> active_rows_1 = io1.get_active_rows();

  if (active_rows_0.size() != active_rows_1.size()) {
    cout << "Size does not match: " << active_rows_0.size()
         << " != " << active_rows_1.size() << endl;
    return false;
  }

  sort(active_rows_0.begin(), active_rows_0.end());
  sort(active_rows_1.begin(), active_rows_1.end());

  for (size_t idx = 0; idx < active_rows_0.size(); idx++) {
    if (active_rows_0[idx] != active_rows_1[idx]) {
      cout << "Difference address in position " << idx << ": "
           << active_rows_0[idx] << " != " << active_rows_1[idx] << endl;
      return false;
    }

    if (io0.get_slice(active_rows_0[idx]) !=
        io1.get_slice(active_rows_1[idx])) {
      cout << "Difference: address=" << active_rows_0[idx] << endl;
      cout << "In file0: " << io0.get_slice(active_rows_0[idx]) << endl;
      cout << "In file1: " << io1.get_slice(active_rows_1[idx]) << endl;
      return false;
    }
  }
  return true;
}
