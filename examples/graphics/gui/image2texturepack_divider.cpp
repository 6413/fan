import fan;

int main(int argc, char** argv) {
  if (argc != 5) {
    fan::print("vertical horizontal input output; example: 2 2 *.png *fte");
    return -1;
  }
  loco_t loco;

  fan::graphics::image_divider_t image_divider;

  std::string image_path = argv[3];

  image_divider.root_image = gloco->image_load(
    image_path.c_str()
  );
  auto& img = gloco->image_get_data(image_divider.root_image);
  image_divider.open_properties.preferred_pack_size = img.size;

  if (image_divider.root_image.iic() == false) {
    auto& img = gloco->image_get_data(image_divider.root_image);
    fan::vec2i divider{std::stoi(argv[1]), std::stoi(argv[2])};
    fan::vec2 uv_size = img.size / divider / img.size;

    image_divider.images.resize(divider.y);
    for (int i = 0; i < divider.y; ++i) {
      image_divider.images[i].resize(divider.x);
      for (int j = 0; j < divider.x; ++j) {
        image_divider.images[i][j] = fan::graphics::image_divider_t::image_t{
          .uv_pos = uv_size * fan::vec2(j, i),
          .uv_size = uv_size,
          .image = image_divider.root_image
        };
      }
    }
    image_divider.clicked_images.resize(divider.multiply());
    for (auto& i : image_divider.clicked_images) {
      i.highlight = 0;
      i.count_index = 0;
    }
  }

  int index = 0;
  for (auto& i : image_divider.images) {
    for (auto& j : i) {
      image_divider.texture_properties.image_name = fan::string("tile") + std::to_string(index);
      image_divider.texture_properties.uv_pos = j.uv_pos;
      image_divider.texture_properties.uv_size = j.uv_size;
      image_divider.e.push_texture(j.image, image_divider.texture_properties);
      ++index;
    }
  }
  image_divider.e.process();
  image_divider.e.save_compiled(argv[4]);
}