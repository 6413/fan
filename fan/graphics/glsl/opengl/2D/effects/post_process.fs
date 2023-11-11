
  #version 330

  in vec2 texture_coordinate;

  in vec4 instance_color;
  in float texture_id;

	layout (location = 0) out vec3 o_color;

	out vec4 out_color;

	uniform vec2 resolution;
	uniform float filter_radius;

  uniform sampler2D _t00;
	uniform sampler2D _t01;

  void main() {
  }
