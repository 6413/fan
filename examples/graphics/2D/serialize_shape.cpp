#include <fan/pch.h>

int main() {
  loco_t loco;//
  fan::time::clock c;
  c.start();

  // serialize
  {
    fan::graphics::rectangle_t r{ {
      .position = 400,
      .size = 50,
      .color = fan::colors::red,
    } };
    // json
    {
      fan::json out;
      for (int i = 0; i < 35000; ++i) {
        fan::json temp;
        fan::graphics::shape_serialize(r, &temp);
        out += temp;
      }

      fan::io::file::write("shapes.json", out.dump(2), std::ios_base::binary);
    }
    // bin
    {
      fan::string out;
      {
        for (int i = 0; i < 35000; ++i) {
          fan::string temp;
          fan::graphics::shape_serialize(r, &temp);
          out += temp;
        }
      }
      fan::print(c.elapsed());
      fan::io::file::write("shapes.bin", out, std::ios_base::binary);
    }
  }

  // deserialize{
  {
    {
      std::string data;
      fan::io::file::read("shapes.json", &data);
      fan::json out = fan::json::parse(data);

      fan::graphics::shape_deserialize_t it;
      loco_t::shape_t shape;
      int i = 0;
      while (it.iterate(out, &shape)) {
        fan::print(i++);
      }
    }
    {
      std::string out;
      fan::io::file::read("shapes.bin", &out);

      fan::graphics::shape_deserialize_t it;
      loco_t::shape_t shape;
      int i = 0;
      while (it.iterate(out, &shape)) {
        fan::print(i++);
      }
    }
  }

  loco.loop([&]() {
    
  });
}