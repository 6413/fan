loco_t& get_loco() {
  return (*OFFSETLESS(this, loco_t, vk));
}
#define loco get_loco()

void shapes_open() {
  loco.shape_functions.resize(loco.shape_functions.size() + 1); // button
  loco.shape_functions.resize(loco.shape_functions.size() + 1); // sprite
  loco.shape_functions.resize(loco.shape_functions.size() + 1); // text
  loco.shape_functions.resize(loco.shape_functions.size() + 1); // hitbox
  loco.shape_functions.resize(loco.shape_functions.size() + 1); // line
  loco.shape_functions.resize(loco.shape_functions.size() + 1); // mark

 /* loco.shape_open<loco_t::rectangle_t>(
    &loco.rectangle,
    "shaders/vulkan/2D/objects/rectangle.vert",
    "shaders/vulkan/2D/objects/rectangle.frag"
  );*/
}

#undef loco