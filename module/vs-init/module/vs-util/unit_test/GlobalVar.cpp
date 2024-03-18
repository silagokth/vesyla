// Copyright (C) 2022 Yu Yang
//
// This file is part of vesyla-suite.
//
// vesyla-suite is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// vesyla-suite is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with vesyla-suite.  If not, see <http://www.gnu.org/licenses/>.

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include "GlobalVar.hpp"
#include <boost/test/unit_test.hpp>

INITIALIZE_EASYLOGGINGPP

using namespace vesyla::util;

BOOST_AUTO_TEST_CASE(read_write) {
  GlobalVar glv;

  int int_inp = 10;
  int int_ref = 10;
  int int_out = 0;
  BOOST_CHECK_EQUAL(glv.puti("int_var", int_inp), true);
  BOOST_CHECK_EQUAL(glv.geti("int_var", int_out), true);
  BOOST_CHECK_EQUAL(int_out, int_ref);
  bool bool_inp = true;
  bool bool_ref = true;
  bool bool_out = false;
  BOOST_CHECK_EQUAL(glv.putb("bool_var", bool_inp), true);
  BOOST_CHECK_EQUAL(glv.getb("bool_var", bool_out), true);
  BOOST_CHECK_EQUAL(bool_out, bool_ref);
  float float_inp = 10.5;
  float float_ref = 10.5;
  float float_out = 0;
  BOOST_CHECK_EQUAL(glv.putf("float_var", float_inp), true);
  BOOST_CHECK_EQUAL(glv.getf("float_var", float_out), true);
  BOOST_CHECK_CLOSE(float_out, float_ref, 0.0001);
  string str_inp = "pass";
  string str_ref = "pass";
  string str_out = "fail";
  BOOST_CHECK_EQUAL(glv.puts("str_var", str_inp), true);
  BOOST_CHECK_EQUAL(glv.gets("str_var", str_out), true);
  BOOST_CHECK_EQUAL(str_out, str_ref);
  GlobalVar::glvar_t var_inp, var_ref, var_out;
  var_inp.type = "integer";
  var_inp.b = true;
  var_inp.i = 2;
  var_inp.f = 5.5;
  strncpy(var_inp.s, "pass", strlen("pass"));
  var_ref = var_inp;
  var_out.type = "string";
  var_out.b = false;
  var_out.i = 3;
  var_out.f = 7.5;
  strncpy(var_out.s, "pass", strlen("fail"));
  BOOST_CHECK_EQUAL(glv.put("var", var_inp), true);
  BOOST_CHECK_EQUAL(glv.get("var", var_out), true);
  BOOST_CHECK_EQUAL(var_out.type, var_ref.type);
  BOOST_CHECK_EQUAL(var_out.b, var_ref.b);
  BOOST_CHECK_EQUAL(var_out.i, var_ref.i);
  BOOST_CHECK_CLOSE(var_out.f, var_ref.f, 0.0001);
  BOOST_CHECK_EQUAL(strncmp(var_out.s, var_ref.s, strlen(var_out.s)), 0);

  GlobalVar glv1;
  int int_ref1 = 10;
  int int_out1 = 0;
  BOOST_CHECK_EQUAL(glv1.geti("int_var", int_out1), true);
  BOOST_CHECK_EQUAL(int_out1, int_ref1);

  glv1.clear();

  BOOST_CHECK_EQUAL(glv.geti("int_var", int_out), false);
  BOOST_CHECK_EQUAL(glv.getb("bool_var", bool_out), false);
  BOOST_CHECK_EQUAL(glv.getf("float_var", float_out), false);
  BOOST_CHECK_EQUAL(glv.gets("str_var", str_out), false);
}

BOOST_AUTO_TEST_CASE(load_store) {
  GlobalVar glv;

  BOOST_CHECK_EQUAL(glv.puti("/int_var/var1", 1), true);
  BOOST_CHECK_EQUAL(glv.puti("/int_var/var2", 2), true);
  BOOST_CHECK_EQUAL(glv.puti("/int_var/var3", 3), true);

  BOOST_CHECK_EQUAL(glv.putf("/float_var/var1", 0.1), true);
  BOOST_CHECK_EQUAL(glv.putf("/float_var/var2", 0.2), true);
  BOOST_CHECK_EQUAL(glv.putf("/float_var/var3", 0.3), true);

  std::set<string> filters = {"/int_var/var1", "/int_var/var2",
                              "/int_var/var3"};
  glv.store_vars("GlobalVar_test_1.json");
  glv.select_and_store_vars_white("GlobalVar_test_1_int.json", filters);
  glv.select_and_store_vars_black("GlobalVar_test_1_float.json", filters);

  int int_out;
  float float_out;

  glv.clear();
  glv.load_vars("GlobalVar_test_1.json");
  BOOST_CHECK_EQUAL(glv.geti("/int_var/var1", int_out), true);
  BOOST_CHECK_EQUAL(int_out, 1);
  BOOST_CHECK_EQUAL(glv.geti("/int_var/var2", int_out), true);
  BOOST_CHECK_EQUAL(int_out, 2);
  BOOST_CHECK_EQUAL(glv.geti("/int_var/var3", int_out), true);
  BOOST_CHECK_EQUAL(int_out, 3);
  BOOST_CHECK_EQUAL(glv.getf("/float_var/var1", float_out), true);
  BOOST_CHECK_CLOSE(float_out, 0.1, 0.0001);
  BOOST_CHECK_EQUAL(glv.getf("/float_var/var2", float_out), true);
  BOOST_CHECK_CLOSE(float_out, 0.2, 0.0001);
  BOOST_CHECK_EQUAL(glv.getf("/float_var/var3", float_out), true);
  BOOST_CHECK_CLOSE(float_out, 0.3, 0.0001);

  glv.clear();
  glv.load_vars("GlobalVar_test_1_int.json");
  BOOST_CHECK_EQUAL(glv.geti("/int_var/var1", int_out), true);
  BOOST_CHECK_EQUAL(int_out, 1);
  BOOST_CHECK_EQUAL(glv.geti("/int_var/var2", int_out), true);
  BOOST_CHECK_EQUAL(int_out, 2);
  BOOST_CHECK_EQUAL(glv.geti("/int_var/var3", int_out), true);
  BOOST_CHECK_EQUAL(int_out, 3);
  BOOST_CHECK_EQUAL(glv.getf("/float_var/var1", float_out), false);
  BOOST_CHECK_EQUAL(glv.getf("/float_var/var2", float_out), false);
  BOOST_CHECK_EQUAL(glv.getf("/float_var/var3", float_out), false);

  glv.clear();
  glv.load_vars("GlobalVar_test_1_float.json");
  BOOST_CHECK_EQUAL(glv.geti("/int_var/var1", int_out), false);
  BOOST_CHECK_EQUAL(glv.geti("/int_var/var2", int_out), false);
  BOOST_CHECK_EQUAL(glv.geti("/int_var/var3", int_out), false);
  BOOST_CHECK_EQUAL(glv.getf("/float_var/var1", float_out), true);
  BOOST_CHECK_CLOSE(float_out, 0.1, 0.0001);
  BOOST_CHECK_EQUAL(glv.getf("/float_var/var2", float_out), true);
  BOOST_CHECK_CLOSE(float_out, 0.2, 0.0001);
  BOOST_CHECK_EQUAL(glv.getf("/float_var/var3", float_out), true);
  BOOST_CHECK_CLOSE(float_out, 0.3, 0.0001);
}
