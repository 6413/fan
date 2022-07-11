R"(
  #version 150

  out vec4 color;

  uniform mat4 view;
  uniform mat4 projection;

  in vec4 instance_color;
  in vec2 tcs;
  in vec4 outline_color;
  in float outline_size;
  in float aspect_ratio;

  void main() {
    color = instance_color;
    
    vec2 p = abs(tcs);
    vec2 border_size = vec2(1.0) - outline_size * vec2(aspect_ratio, 1);

    if (p.x > border_size.x) {
      color = outline_color;
    }
    if (p.y > border_size.y) {
      color = outline_color;
    }
  }
)"