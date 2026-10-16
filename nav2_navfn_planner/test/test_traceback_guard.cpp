// Copyright (c) 2026, Vincent Samy
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <cmath>
#include <vector>

#include "gtest/gtest.h"
#include "nav2_navfn_planner/navfn.hpp"

namespace
{

constexpr int kNx = 100;
constexpr int kNy = 120;
constexpr double kResolution = 0.05;

void fill(std::vector<unsigned char> & cost, int x0, int y0, int x1, int y1, unsigned char value)
{
  for (int y = std::max(0, y0); y <= std::min(kNy - 1, y1); ++y) {
    for (int x = std::max(0, x0); x <= std::min(kNx - 1, x1); ++x) {
      cost[y * kNx + x] = value;
    }
  }
}

// Inflate lethal cells like nav2_costmap_2d: inscribed cost up to the inscribed
// radius, then exponential decay up to the inflation radius.
void inflate(std::vector<unsigned char> & cost, double inscribed, double radius, double scale)
{
  std::vector<unsigned char> out = cost;
  const int reach = static_cast<int>(std::ceil(radius / kResolution));
  for (int y = 0; y < kNy; ++y) {
    for (int x = 0; x < kNx; ++x) {
      if (cost[y * kNx + x] != 254) {
        continue;
      }
      for (int dy = -reach; dy <= reach; ++dy) {
        for (int dx = -reach; dx <= reach; ++dx) {
          const int xx = x + dx;
          const int yy = y + dy;
          if (xx < 0 || yy < 0 || xx >= kNx || yy >= kNy) {
            continue;
          }
          const double d = std::hypot(dx, dy) * kResolution;
          if (d > radius) {
            continue;
          }
          const unsigned char c = (d <= inscribed) ? 253 :
            static_cast<unsigned char>(252.0 * std::exp(-scale * (d - inscribed)));
          unsigned char & o = out[yy * kNx + xx];
          if (o != 254 && c > o) {
            o = c;
          }
        }
      }
    }
  }
  cost = out;
}

// A 5 m x 6 m floor plan: outer wall, a 2 m corridor along y with doors every
// 6 m, room partitions and three pillars. The A* potential field computed on
// it from (13, 49) to (56, 81) traps the plain gradient descent.
std::vector<unsigned char> makeTrapMap()
{
  std::vector<unsigned char> cost(kNx * kNy, 0);
  fill(cost, 0, 0, kNx - 1, 1, 254);
  fill(cost, 0, kNy - 2, kNx - 1, kNy - 1, 254);
  fill(cost, 0, 0, 1, kNy - 1, 254);
  fill(cost, kNx - 2, 0, kNx - 1, kNy - 1, 254);
  const int corridor_width = 40;
  const int corridor_x = kNx / 2 - corridor_width / 2;
  const int door = 24;
  const int pitch = 120;
  for (int y = 2; y < kNy - 2; ++y) {
    if ((y % pitch) >= door) {
      cost[y * kNx + corridor_x - 1] = 254;
      cost[y * kNx + corridor_x + corridor_width] = 254;
    }
  }
  for (int y = pitch; y < kNy - 2; y += pitch) {
    fill(cost, 2, y, corridor_x - 2, y, 254);
    fill(cost, corridor_x + corridor_width + 1, y, kNx - 3, y, 254);
  }
  fill(cost, 84, 82, 90, 88, 254);
  fill(cost, 43, 34, 45, 36, 254);
  fill(cost, 20, 23, 24, 27, 254);
  inflate(cost, 0.30, 0.80, 3.0);
  return cost;
}

}  // namespace

TEST(NavFnTracebackTest, PartiallyRelaxedAstarFieldDoesNotTrapPathExtraction)
{
  std::vector<unsigned char> costmap = makeTrapMap();
  // NavFn propagates the potential from its goal (the robot pose, cleared like
  // NavfnPlanner does) and extracts the path from its start (the target pose).
  int robot[2] = {13, 49};
  int target[2] = {56, 81};
  costmap[robot[1] * kNx + robot[0]] = 0;

  nav2_navfn_planner::NavFn navfn(kNx, kNy);
  navfn.setCostmap(costmap.data(), true, false);
  navfn.setStart(target);
  navfn.setGoal(robot);
  ASSERT_TRUE(navfn.calcNavFnAstar([]() {return false;}));
  ASSERT_LT(navfn.potarr[target[1] * kNx + target[0]], POT_HIGH);

  const int budget = 4 * std::max(kNx, kNy);

  // The plain descent loops on this field whatever the budget.
  EXPECT_EQ(navfn.tracePath(budget, nullptr, false), -1);
  EXPECT_EQ(navfn.tracePath(budget * 50, nullptr, false), -1);

  // calcPath falls back to the guarded descent and reaches the robot cell.
  const int len = navfn.calcPath(budget);
  ASSERT_GT(len, 0);
  EXPECT_LE(len, budget);
  EXPECT_FLOAT_EQ(navfn.pathx[len - 1], static_cast<float>(robot[0]));
  EXPECT_FLOAT_EQ(navfn.pathy[len - 1], static_cast<float>(robot[1]));
}

TEST(NavFnTracebackTest, PathExtractedByThePlainDescentIsUnchanged)
{
  std::vector<unsigned char> costmap = makeTrapMap();
  int robot[2] = {13, 49};
  int target[2] = {56, 81};
  costmap[robot[1] * kNx + robot[0]] = 0;

  nav2_navfn_planner::NavFn navfn(kNx, kNy);
  navfn.setCostmap(costmap.data(), true, false);
  navfn.setStart(target);
  navfn.setGoal(robot);
  ASSERT_TRUE(navfn.calcNavFnDijkstra([]() {return false;}, true));

  const int budget = 4 * std::max(kNx, kNy);
  const int plain_len = navfn.tracePath(budget, nullptr, false);
  ASSERT_GT(plain_len, 0);
  const std::vector<float> plain_x(navfn.pathx, navfn.pathx + plain_len);
  const std::vector<float> plain_y(navfn.pathy, navfn.pathy + plain_len);

  ASSERT_EQ(navfn.calcPath(budget), plain_len);
  for (int i = 0; i < plain_len; ++i) {
    EXPECT_EQ(navfn.pathx[i], plain_x[i]) << "x differs at point " << i;
    EXPECT_EQ(navfn.pathy[i], plain_y[i]) << "y differs at point " << i;
  }
}
