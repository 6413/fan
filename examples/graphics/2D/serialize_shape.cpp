#include <fan/pch.h>

int main() {
  loco_t loco;//
  fan::time::clock c;
  

  // serialize
  {
    fan::graphics::rectangle_t r{ {
      .position = 400,
      .size = 50,
      .color = fan::colors::red,
    } };
    // json
    {
      c.start();
      fan::json out;
      for (int i = 0; i < 35000; ++i) {
        fan::json temp;
        fan::graphics::shape_serialize(r, &temp);
        out += temp;
      }

      fan::io::file::write("shapes.json", out.dump(2), std::ios_base::binary);
      fan::print("json serialize time:", c.elapsed());
    }
    // bin
    {
      c.start();
      std::vector<uint8_t> out;
      {
        for (int i = 0; i < 35000; ++i) {
          std::vector<uint8_t> temp;
          fan::graphics::shape_serialize(r, &temp);
          out.insert(out.end(), temp.begin(), temp.end());
        }
      }
      fan::io::file::write("shapes.bin", out, std::ios_base::binary);
      fan::print("bin serialize time:", c.elapsed());
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
        //fan::print(i++);
      }
    }
  }

  loco.loop([&]() {
    
  });
}