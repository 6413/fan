#version 330

layout (location = 0) out vec4 o_attachment0;

in vec2 texture_coordinate;
in vec2 instance_position;
in vec2 instance_size;
in vec2 grid_size;
in vec4 instance_color;
in vec3 frag_position;

uniform vec2 matrix_size;
uniform vec4 viewport;
uniform vec2 window_size;
uniform vec2 camera_position;

uniform float camera_zoom;

void main() {
  vec2 viewport_position = viewport.xy;
  vec2 viewport_size = viewport.zw;
  vec2 grid_thickness = vec2(max(1.0, camera_zoom*2.f)); // user input

  // spacing per cell based on full size
  vec2 full_size = instance_size / matrix_size * viewport_size;
  vec2 d = full_size / grid_size;

  // only for coordinates which are y+ down for "1.0 - viewport_position.y"
  vec2 fragcoord = vec2(camera_position.x, camera_position.y) / matrix_size * viewport_size +
  vec2(1.0 - viewport_position.x, 1.0 - viewport_position.y) + 
  vec2(gl_FragCoord.x, window_size.y - gl_FragCoord.y) - viewport_size / 2;  //-size / 2, +size / 2 == - viewport_size / 2. 0, +size == - 0

  // relative position from center
  vec2 center = instance_position / matrix_size * viewport_size;
  vec2 rpos = fragcoord - center;

  float m0 = mod(rpos.x, d.x); // modulo0
  float m1 = mod(rpos.y, d.y); // modulo1
  float thickness_x = grid_thickness.x;
  float thickness_y = grid_thickness.y;
  float aa_x = fwidth(m0);
  float aa_y = fwidth(m1);

  float line_x = 1.0 - smoothstep(thickness_x - aa_x, thickness_x + aa_x, m0);
  float line_y = 1.0 - smoothstep(thickness_y - aa_y, thickness_y + aa_y, m1);
  float line = max(line_x, line_y);
  o_attachment0 = instance_color * line;
  /*if(m0 <= grid_thickness.x || m1 <= grid_thickness.y){
    o_attachment0 = instance_color;
  }
  else{
    discard;
  } */
}
