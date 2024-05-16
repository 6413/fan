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

void main() {
  vec2 viewport_position = viewport.xy;
  vec2 viewport_size = viewport.zw;
  vec2 grid_thickness = vec2(1.5); // user input
  vec2 d = instance_size / matrix_size * viewport_size;
  d /= grid_size;
                                                  // only for coordinates which are y+ down for "1.0 - viewport_position.y"
  vec2 fragcoord = vec2(camera_position.x, camera_position.y) / matrix_size * viewport_size +
  vec2(1.0 - viewport_position.x, 1.0 - viewport_position.y) + 
  vec2(gl_FragCoord.x, window_size.y - gl_FragCoord.y) - viewport_size / 2; //-size / 2, +size / 2 == - viewport_size / 2. 0, +size == - 0
  vec2 half_size = instance_size / matrix_size * viewport_size;
  vec2 rpos = fragcoord; // relative position
  vec2 stlfp = (instance_position - instance_size) / matrix_size * viewport_size; // shape top left fragment position 
  rpos = rpos - stlfp;
  vec2 adder = (half_size + grid_thickness / 2.) / half_size;
  rpos *= adder;

  float m0 = mod(rpos.x, d.x); // modulo0
  float m1 = mod(rpos.y, d.y); // modulo1
  if(m0 <= grid_thickness.x || m1 <= grid_thickness.y){
    o_attachment0 = instance_color;
  }
  else{
    discard;
  }
}
