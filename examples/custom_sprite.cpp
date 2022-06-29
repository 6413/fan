// Creates window, opengl context and renders a rectangle

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#define FAN_INCLUDE_PATH C:/libs/fan/include
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#include _FAN_PATH(graphics/graphics.h)

using id_holder_t = bll_t<uint32_t>;

struct pile_t {
  fan::opengl::matrices_t matrices;
  fan::window_t window;
  fan::opengl::context_t context;
  id_holder_t ids;
};

struct sprite_t : fan_2d::graphics::sprite_t<pile_t*, uint32_t> {

  using inherit_t = fan_2d::graphics::sprite_t<pile_t*, uint32_t>;

  void open(fan::opengl::context_t* context, move_cb_t move_cb_, const user_global_data_t& gd) {
    fan_2d::graphics::sprite_t<pile_t*, uint32_t>::open(context, move_cb_, gd);
    set_vertex(
      context,
#include _FAN_PATH(graphics/glsl/opengl/2D/objects/sprite.vs)
    );
    set_fragment(context,
      R"(
        #version 330

        in vec2 texture_coordinate;

        in vec4 instance_color;

        out vec4 o_color;

        uniform sampler2D texture_sampler;

        uniform float input0;
        uniform float input1;

        vec2 hash( vec2 p ) // replace this by something better
        {
	        p = vec2( dot(p,vec2(127.1,311.7)), dot(p,vec2(269.5,183.3)) );
	        return -1.0 + 2.0*fract(sin(p)*43758.5453123);
        }

        float noise( in vec2 p )
        {
            const float K1 = 0.366025404; // (sqrt(3)-1)/2;
            const float K2 = 0.211324865; // (3-sqrt(3))/6;

	        vec2  i = floor( p + (p.x+p.y)*K1 );
            vec2  a = p - i + (i.x+i.y)*K2;
            float m = step(a.y,a.x); 
            vec2  o = vec2(m,1.0-m);
            vec2  b = a - o + K2;
	        vec2  c = a - 1.0 + 2.0*K2;
            vec3  h = max( 0.5-vec3(dot(a,a), dot(b,b), dot(c,c) ), 0.0 );
	        vec3  n = h*h*h*h*vec3( dot(a,hash(i+0.0)), dot(b,hash(i+o)), dot(c,hash(i+1.0)));
            return dot( n, vec3(70.0) );
        }

        float rand(vec2 co){
        return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453);
        }

        void main() {

        vec2 iResolution = vec2(800, 600);

        vec2 p = gl_FragCoord.xy / iResolution.xy;

	        vec2 uv = p*vec2(iResolution.x/iResolution.y,1.0) + input0;
	
	        float f = 0.0;

	        uv *= 5.0;
              mat2 m = mat2( 1.6,  1.2, -1.2,  1.6 );
	        f  = 0.5000*noise( uv ); uv = m*uv;
	        f += 0.2500*noise( uv ); uv = m*uv;
	        f += 0.1250*noise( uv ); uv = m*uv;
	        f += 0.0625*noise( uv ); uv = m*uv;

	        f = 0.5 + 0.5*f;

          o_color = texture(texture_sampler, texture_coordinate) * instance_color;
	        float b = o_color.b;
	        if (o_color.b > 0) {
		        if (f > 0.5) {
			        o_color.g = 0.501960784313725;
			        o_color.b = 0;
		        }
	        f += 0.3500*noise( uv * (f * input1)); uv = m*uv;
		        float s = f * (f / 1.1);
		        if (s > 0.5) {
			        o_color.g -= 1;
			        o_color.b = b;
		        }
	        }
        }
      )"
    );
    sprite_t::compile(context);
    inputs = {};
    sprite_t::set_draw_cb(context, sprite_t::draw_cb, &inputs);
  }

  struct input_t {
    f32_t input0;
    f32_t input1;
  }inputs;

  static void draw_cb(fan::opengl::context_t* context, sprite_t::inherit_t* sprite, void* userptr) {
    sprite_t::input_t& input = *(sprite_t::input_t*)userptr;
    sprite->m_shader.set_float(context, "input0", input.input0);
    sprite->m_shader.set_float(context, "input1", input.input1);
  }

};

void cb(sprite_t* l, uint32_t src, uint32_t dst, uint32_t* p) {
  l->user_global_data->ids[*p] = dst;
}

int main() {

  pile_t pile;

  pile.window.open();

  pile.context.init();
  pile.context.bind_to_window(&pile.window);
  pile.context.set_viewport(0, pile.window.get_size());
  pile.window.add_resize_callback(&pile, [](fan::window_t*, const fan::vec2i& size, void* userptr) {
    pile_t* pile = (pile_t*)userptr;

    pile->context.set_viewport(0, size);

    fan::vec2 window_size = pile->window.get_size();
    fan::vec2 ratio = window_size / window_size.max();
    std::swap(ratio.x, ratio.y);
    pile->matrices.set_ortho(&pile->context, fan::vec2(-1, 1) * ratio.x, fan::vec2(-1, 1) * ratio.y);
    });

  pile.matrices.open();


  pile.ids.open();

  sprite_t s;
  s.open(&pile.context, (sprite_t::move_cb_t)cb, &pile);
  s.bind_matrices(&pile.context, &pile.matrices);
  s.enable_draw(&pile.context);

  sprite_t::properties_t p;

  fan::opengl::image_t::load_properties_t lp;
  lp.filter = fan::opengl::GL_LINEAR;
  p.image.load(&pile.context, "images/planet.webp", lp);
  p.size = fan::cast<f32_t>(p.image.size) / pile.window.get_size();

  p.position = fan::random::vec2(0, 0);
  for (uint32_t i = 0; i < 1; i++) {
    uint32_t it = pile.ids.push_back(s.push_back(&pile.context, p));
    s.set_user_instance_data(&pile.context, pile.ids[it], it);

    /* EXAMPLE ERASE
    s.erase(&pile.context, pile.ids[it]);
    pile.ids.erase(it);
    */
  }

  fan::vec2 window_size = pile.window.get_size();
  fan::vec2 ratio = window_size / window_size.max();
  std::swap(ratio.x, ratio.y);
  pile.matrices.set_ortho(&pile.context, fan::vec2(-1, 1), fan::vec2(-1, 1));

  pile.window.add_keys_callback(&s, [](fan::window_t*, uint16_t key, fan::key_state key_state, void* user_ptr) {

    if (key_state != fan::key_state::press) {
      return;
    }

    sprite_t& pile = *(sprite_t*)user_ptr;

    switch (key) {
      case fan::mouse_scroll_up: {
        pile.inputs.input0 += 0.05;
        break;
      }
      case fan::mouse_scroll_down: {
        pile.inputs.input0 -= 0.05;
        break;
      }
      case fan::key_up: {
        pile.inputs.input1 += 0.05;
        break;
      }
      case fan::key_down: {
        pile.inputs.input1 -= 0.05;
        break;
      }
    }
    });

  while (1) {

    uint32_t window_event = pile.window.handle_events();
    if (window_event & fan::window_t::events::close) {
      pile.window.close();
      break;
    }

    pile.context.process();
    pile.context.render(&pile.window);
  }

  return 0;
}