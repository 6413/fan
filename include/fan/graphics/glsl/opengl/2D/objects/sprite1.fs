
  #version 330
  in vec2 texture_coordinate;
  in vec4 instance_color;
  out vec4 o_color;
  uniform sampler2D _tt0;
  uniform sampler2D _tt1;
  uniform vec2 viewport_size;
  void main() {
    vec2 flipped_y = vec2(texture_coordinate.x, texture_coordinate.y);
    vec4 texture_color = texture(_tt0, flipped_y);
    vec2 p = gl_FragCoord.xy / viewport_size;
    vec4 light = texture(_tt1, vec2(p.x, 1.0 - p.y));
    o_color = texture_color * vec4(light.xyz, 1) * instance_color;
  }
