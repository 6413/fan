
  #version 330

  layout (location = 0) out vec4 o_attachment0;

  uniform mat4 view;
  uniform mat4 projection;

  in vec4 instance_color;
  in vec2 tcs;
  in vec4 outline_color;
  in float outline_size;
  in float aspect_ratio;

  void main() {
    o_attachment0 = instance_color;
    
    vec2 p = abs(tcs);
    vec2 border_size = vec2(1.0) - outline_size * vec2(aspect_ratio, 1);

    if (p.x > border_size.x) {
      o_attachment0 = outline_color;
    }
    if (p.y > border_size.y) {
      o_attachment0 = outline_color;
    }
  }
