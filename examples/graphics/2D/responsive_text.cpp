#include fan_pch

struct pile_t {

  static constexpr fan::vec2 ortho_x = fan::vec2(0, 800);
  static constexpr fan::vec2 ortho_y = fan::vec2(0, 800);

  pile_t() {
    auto ws = loco.window.get_size();
    auto viewport_size = fan::vec2(ws.x, ws.y);
    fan::vec2 ratio = viewport_size / viewport_size.max();
   // std::swap(ratio.x, ratio.y);
    loco.open_camera(
      &camera,
      ortho_x * ratio.x,
      ortho_y * ratio.y
    );
    loco.window.add_resize_callback([&](const fan::window_t::resize_cb_data_t& d) {
      viewport.set(0, fan::vec2(d.size.x, d.size.y), d.size);
    });
    loco.open_viewport(&viewport, fan::vec2(0, 0), viewport_size);
  }

  loco_t loco;
  loco_t::camera_t camera;
  loco_t::viewport_t viewport;
};

struct sprite_responsive_t : loco_t::shape_t {
  using properties_t = loco_t::shapes_t::responsive_text_t::properties_t;

  sprite_responsive_t(loco_t::shapes_t::responsive_text_t::properties_t rp, const loco_t::shapes_t::sprite_t::properties_t& p) {
    rp.position = p.position;
    rp.position.z += 1;
    rp.size = p.size;
    rp.camera = p.camera;
    rp.viewport = p.viewport;
    *(loco_t::shape_t*)this = rp;
    base = p;
  }

  void set_size(const fan::vec2& s) {
    base.set_size(s);
    gloco->shapes.responsive_text.set_size(*(loco_t::shape_t*)this, s);
   // responsive_text.set_size(s);
  }

  loco_t::shape_t base;
};

int main() {
  pile_t* pile = new pile_t;

  loco_t::image_t image;
  image.load("images/brick.webp");

  loco_t::shapes_t::sprite_t::properties_t pp;

  pp.size = fan::vec2(1);
  pp.camera = &pile->camera;
  pp.viewport = &pile->viewport;  
  pp.image = &image;
  
  pp.position = fan::vec2(400, 100);
  pp.size = fan::vec2(300, 100);
  pp.color.a = 1;
  pp.blending = true;

  sprite_responsive_t::properties_t rtp;
  rtp.camera = &pile->camera;
  rtp.viewport = &pile->viewport;
  //rtp.alignment.alignment = loco_t::responsive_text_t::align_e::center;
  
  rtp.position = fan::vec3(400, 400, 100);
  rtp.text = "222";
  rtp.line_limit = 1;
  rtp.letter_size_y_multipler = 1.f / 1;
  rtp.size = fan::vec2(300, 100);

  //rtp.
  sprite_responsive_t shape(rtp, pp);

  sprite_responsive_t shape2(rtp, pp);
  
  fan::time::clock c;
  c.start();
  f32_t advance = 0;
 /* shape.append_letter(L'g', true);
  shape.append_letter(L'g', true);
  shape.append_letter(L'g', true);
  shape.append_letter(L'2');
  shape.append_letter(L')');*/
  /*while (shape.append_letter(L'2')) {

  }*/
  
//  fan::print("durumssalsa", gloco->responsive_text.line_list[gloco->responsive_text.tlist[shape.gdp4()].LineStartNR].total_width);

  fan::print(c.elapsed());


  //sprite_responsive_t responsive_box(tp, pp);

  /*{
    uint32_t i = 0;
    while (responsive_box.does_text_fit(std::to_string(i))) {
      responsive_box.push_back(std::to_string(i));
      i++;
    }
  }*/


  //fan::vec2
//  responsive_box.set_size(fan::vec2(1, 100));

 /* gloco->window.add_keys_callback([&](const auto& d) {
    if (d.key == fan::key_up) {
      pp.size.x += 1;
      responsive_box.set_size(pp.size);
    }
    if (d.key == fan::key_down) {
      pp.size.x -= 1;
      responsive_box.set_size(pp.size);
    }
  });*/

  f32_t deltaer = 0;
  pile->loco.loop([&] {
    deltaer += pile->loco.get_delta_time() * 0.2    ;
    shape.set_size(fan::vec2(std::abs(sin(deltaer)) * 100 + 25));
    //r0.set_position(pile->loco.get_mouse_position(pile->camera, pile->viewport));
  });

  return 0;
}