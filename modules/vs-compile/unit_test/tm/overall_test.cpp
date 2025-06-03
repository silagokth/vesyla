#include "tm/Solver.hpp"
#include "tm/TimingModel.hpp"
#include <gtest/gtest.h>

TEST(tm, overall_test_1) {

  vesyla::tm::TimingModel tm;
  tm.from_string(
      R"(
  operation route0r e0
  operation input_r R<2,0>(R<2,0>(e0))
  operation input_w R<4,0>(e0)
  operation read_ab R<4,0>(e0)
  operation route1wr e0
  operation write_a R<2,t1>(e0)
  operation write_b R<2,t1>(e0)
  operation swb e0
  operation read_a_seq R<32,0>(e0)
  operation read_b_seq R<32,0>(e0)
  operation write_c_seq R<32,0>(e0)
  operation compute e0
  operation read_c R<2,0>(e0)
  operation route2w e0
  operation write_c R<2,0>(e0)
  operation output_r R<2,0>(e0)
  operation output_w R<2,0>(e0)
  
  constraint linear ( input_r == input_w )
  constraint linear ( input_w < read_ab )
  constraint linear ( route0r < read_ab )
  constraint linear ( route1wr < write_a )
  constraint linear ( route1wr < write_b )
  constraint linear ( read_ab.e0[0] == write_a.e0[0] )
  constraint linear ( read_ab.e0[1] == write_b.e0[0] )
  constraint linear ( read_ab.e0[2] == write_a.e0[1] )
  constraint linear ( read_ab.e0[3] == write_b.e0[1] )
  constraint linear ( write_a < read_a_seq )
  constraint linear ( write_b < read_b_seq )
  constraint linear ( swb < read_a_seq )
  constraint linear ( read_a_seq == read_b_seq )
  constraint linear ( read_a_seq + 1 > compute )
  constraint linear ( write_c_seq == read_a_seq + 1 )
  constraint linear ( compute != route1wr )
  constraint linear ( compute != swb )
  constraint linear ( read_c.e0[0] > write_c_seq.e0[15] )
  constraint linear ( read_c.e0[1] > write_c_seq.e0[31] )
  constraint linear ( write_c == read_c )
  constraint linear ( output_r > write_c )
  constraint linear ( output_r == output_w )
  constraint linear route1wr != compute
      )");
  vesyla::tm::Solver solver("/tmp");
  unordered_map<string, string> result = solver.solve(tm);

  EXPECT_NE(result.find("write_a"), result.end());
  EXPECT_EQ(result["write_a"], "1");
  EXPECT_NE(result.find("input_w"), result.end());
  EXPECT_EQ(result["input_w"], "0");
  EXPECT_NE(result.find("read_ab"), result.end());
  EXPECT_EQ(result["read_ab"], "1");
  EXPECT_NE(result.find("route0r"), result.end());
  EXPECT_EQ(result["route0r"], "0");
  EXPECT_NE(result.find("output_r"), result.end());
  EXPECT_EQ(result["output_r"], "36");
  EXPECT_NE(result.find("write_c"), result.end());
  EXPECT_EQ(result["write_c"], "35");
  EXPECT_NE(result.find("output_w"), result.end());
  EXPECT_EQ(result["output_w"], "36");
  EXPECT_NE(result.find("swb"), result.end());
  EXPECT_EQ(result["swb"], "0");
  EXPECT_NE(result.find("read_b_seq"), result.end());
  EXPECT_EQ(result["read_b_seq"], "3");
  EXPECT_NE(result.find("total_latency"), result.end());
  EXPECT_EQ(result["total_latency"], "38");
  EXPECT_NE(result.find("route1wr"), result.end());
  EXPECT_EQ(result["route1wr"], "0");
  EXPECT_NE(result.find("write_b"), result.end());
  EXPECT_EQ(result["write_b"], "2");
  EXPECT_NE(result.find("input_r"), result.end());
  EXPECT_EQ(result["input_r"], "0");
  EXPECT_NE(result.find("write_c_seq"), result.end());
  EXPECT_EQ(result["write_c_seq"], "4");
  EXPECT_NE(result.find("compute"), result.end());
  EXPECT_EQ(result["compute"], "1");
  EXPECT_NE(result.find("read_c"), result.end());
  EXPECT_EQ(result["read_c"], "35");
  EXPECT_NE(result.find("route2w"), result.end());
  EXPECT_EQ(result["route2w"], "0");
  EXPECT_NE(result.find("t1"), result.end());
  EXPECT_EQ(result["t1"], "1");
  EXPECT_NE(result.find("read_a_seq"), result.end());
  EXPECT_EQ(result["read_a_seq"], "3");
}

TEST(tm, overall_test_2) {
  vesyla::tm::TimingModel tm;
  tm.from_string(
      R"(
    operation read_b R<5, t3>(R<3, t2>(e0))
 operation read_a R<5, t1>(R<3, t0>(e0))
    constraint linear read_a==read_b  
    constraint linear read_a.e0[0]== read_b.e0  [0]
    constraint linear read_a.e0[1]==read_b.e0 [1]            
    constraint linear read_a.e0[1][2]==read_b.e0  [1] [2]  
)");
  vesyla::tm::Solver solver("/tmp");
  unordered_map<string, string> result = solver.solve(tm);
  EXPECT_NE(result.find("read_a"), result.end());
  EXPECT_EQ(result["read_a"], "0");
  EXPECT_NE(result.find("read_b"), result.end());
  EXPECT_EQ(result["read_b"], "0");
  EXPECT_NE(result.find("t0"), result.end());
  EXPECT_EQ(result["t0"], "0");
  EXPECT_NE(result.find("t1"), result.end());
  EXPECT_EQ(result["t1"], "0");
  EXPECT_NE(result.find("t2"), result.end());
  EXPECT_EQ(result["t2"], "0");
  EXPECT_NE(result.find("t3"), result.end());
  EXPECT_EQ(result["t3"], "0");
  EXPECT_NE(result.find("total_latency"), result.end());
  EXPECT_EQ(result["total_latency"], "15");
}
