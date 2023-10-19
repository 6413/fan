// rectangle text button using loco
#include <WITCH/WITCH.h>

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#ifndef FAN_INCLUDE_PATH
#define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#define fan_debug 1
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

//#define loco_vulkan

#define loco_window
#define loco_context

//#define loco_post_process
#define loco_rectangle
#define loco_button
#include _FAN_PATH(graphics/loco.h)

int main() {
  fan::trees::split_tree_t qt;
  qt.open(0.5, 0.5, 1);
  static constexpr f32_t n = 3;
  std::vector<fan::trees::split_tree_t::path_t> filler;

  enum direction_e{
    horizontal,
    vertical
  };

  int dir = direction_e::vertical, split_side = 0;
  auto new_path = qt.insert(filler, dir, split_side);

  dir = horizontal;
  filler = new_path;
  auto new_path4 = qt.insert(filler, dir, 0);

  filler = new_path;
  dir = vertical;
  auto new_path2 = qt.insert(filler, dir, 1);

  dir = horizontal;
  filler = new_path2;
  auto new_path3 = qt.insert(filler, dir, 1);

  //filler = new_path3;
  //auto new_path4 = qt.insert(filler, dir, 1);


  //filler = new_path3;
  //auto new_path4 =  qt.insert(filler, 0, 0);

  //filler = new_path2;
  //auto new_path5 = qt.insert(filler, dir, 1);

  //filler = new_path5;
  //auto new_path6 = qt.insert(filler, 1, 0);

  //qt.direction = 1; 
  //qt.insert();


  //p = ps[3];
  //qt.direction = 1;
  //qt.insert(p);

  loco_t loco;

  std::vector<loco_t::simple_rectangle_t> rectangles;
  std::vector<loco_t::simple_rectangle_t> points;

  fan::trees::split_tree_t* qtp = &qt;

  int index = 0;
  int width_count = 0, height_count = 0;

  //auto l = [&rectangles, &index, &points, &width_count, &height_count](const auto& l, fan::trees::split_tree_t* qtp, int vertical_depth, int horizontal_depth) -> void {
  //  if (!qtp) {
  //    return;
  //  }

  //  /*if (qtp->direction == 0 && qtp->direction != -1) {
  //    y++;
  //  }
  //  else if (qtp->direction != -1) {
  //    width_count++;
  //  }*/

  //  if (vertical_depth != 0) {
  //    width_count = std::max(width_count, vertical_depth);
  //  }
  //  if (horizontal_depth != 0) {
  //    height_count = std::max(height_count, horizontal_depth);
  //  }

  //  if (qtp->divided) {
  //    if (qtp->direction == 0) {
  //      int horizontal_child_depth = horizontal_depth + 1;
  //      l(l, qtp->horizontal[0], tree_depth + 1, vertical_depth, horizontal_child_depth);
  //      l(l, qtp->horizontal[1], tree_depth + 1, vertical_depth, horizontal_child_depth);
  //    }
  //    else {
  //      
  //      int vertical_child_depth = vertical_depth + 1;
  //      l(l, qtp->vertical[0], tree_depth + 1, vertical_child_depth, horizontal_depth);
  //      l(l, qtp->vertical[1], tree_depth + 1, vertical_child_depth, horizontal_depth);
  //    }
  //  }
  //  else {

  //  }
  //  };

  {
    auto l2 = [&rectangles, &index, &points, &width_count, &height_count](const auto& l, int dir, int side, fan::trees::split_tree_t* qtp, fan::trees::split_tree_t* other_node, fan::trees::split_tree_t* prev, int tree_depth, int vertical_depth, int horizontal_depth) -> void {
      if (!qtp) {
        return;
      }

      if (qtp->direction == 1 && qtp->direction != -1) {
          width_count++;
      }

      if (qtp->divided) {
        int horizontal_child_depth = horizontal_depth + 1;
        int vertical_child_depth = vertical_depth + 1;
        l(l, 0, 0, qtp->horizontal[0], qtp->horizontal[1], qtp, tree_depth + 1, vertical_depth, horizontal_child_depth);
        l(l, 0, 1, qtp->horizontal[1], qtp->horizontal[0], qtp, tree_depth + 1, vertical_depth, horizontal_child_depth);
        l(l, 1, 0, qtp->vertical[0], qtp->vertical[1], qtp, tree_depth + 1, vertical_child_depth, horizontal_depth);
        l(l, 1, 1, qtp->vertical[1], qtp->vertical[0], qtp, tree_depth + 1, vertical_child_depth, horizontal_depth);
      }
      else {

      }
      };

    int initial_vertical_depth = 0;
    int initial_horizontal_depth = 0;
    l2(l2, 0, 0, qtp, 0, 0, 0, initial_vertical_depth, initial_horizontal_depth);
  }

  auto l2 = [&rectangles, &index, &points, &width_count, &height_count](const auto& l, int dir, int side, fan::trees::split_tree_t* qtp, fan::trees::split_tree_t* other_node, fan::trees::split_tree_t* prev, int tree_depth, int vertical_depth, int horizontal_depth) -> void {
    if (!qtp) {
      return;
    }

    // need to also add rectangles here
    if (qtp->divided) {
      int horizontal_child_depth = horizontal_depth + 1;
      l(l, 0, 0, qtp->horizontal[0], qtp->horizontal[1], qtp, tree_depth + 1, vertical_depth, horizontal_child_depth);
      l(l, 0, 1, qtp->horizontal[1], qtp->horizontal[0], qtp, tree_depth + 1, vertical_depth, horizontal_child_depth);
      int vertical_child_depth = vertical_depth + 1;
      l(l, 1, 0, qtp->vertical[0], qtp->vertical[1],     qtp, tree_depth + 1, vertical_child_depth, horizontal_depth);
      l(l, 1, 1, qtp->vertical[1], qtp->vertical[0],     qtp, tree_depth + 1, vertical_child_depth, horizontal_depth);
    }
    else {
      fan::vec2 p = qtp->position * 800.f;

      int sumx = 1, sumy = 1;

      if (prev) {
        if (prev->direction == 0) {
          sumy = -1;
        }
        else {
          sumx = -1;
        }
      }

      fan::print_struct(*qtp);
      float sectionSizex = 800.f / width_count;
      float centerPosx = ((tree_depth - 1) * sectionSizex) + (sectionSizex / 2);

      float sectionSizey = 800.f / (horizontal_depth + 1);
      float centerPosy = ((side) * sectionSizey) + (sectionSizey / 2);

      rectangles.push_back(loco_t::simple_rectangle_t{ {
        .position = fan::vec3((fan::vec2(centerPosx, p.y) / 800.f) * 2 - 1, index++),
        .size = fan::vec2(sectionSizex / 2, qtp->boundary.y * 800) / 800.f * 2, // because coordinate is -1 -> 1 so * 2 when viewport size is 0-1
        .color = fan::random::color() - fan::vec4(0, 0, 0, 0.5)
        /*fan::color::hsv((float)index / n * 360.f, 100, 100)*/,
      .blending = true
      } });
    }
  };

  int initial_vertical_depth = 0;
  int initial_horizontal_depth = 0;
  l2(l2, 0, 0, qtp, 0, 0, 0, initial_vertical_depth, initial_horizontal_depth);


  fan::print(width_count, height_count);
  
  std::cout << std::format("{}, {}", 4, 3);

 /* std::ostringstream oss;
  oss << fan::vec2();
  fan::print_format("point: {}, index: {}", fan::vec2(), 3);*/
  //fmt::print("point: {}, index {}", fan::vec2(), 3);
  //std::ostream::operator<<(int());

  loco.loop([&] {

  });

  return 0;
}
