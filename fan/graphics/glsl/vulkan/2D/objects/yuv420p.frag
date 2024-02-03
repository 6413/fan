#version 420

layout(location = 0) out vec4 o_color;
layout(location = 1) out vec4 rcolor;

layout(binding = 2) uniform sampler2D _t[16];

layout(location = 0) in vec4 instance_color;
layout(location = 1) in vec2 texture_coordinate;

void main() {
  rcolor = vec4(0);
	vec3 yuv;

	yuv.x = texture(_t[0], vec2(texture_coordinate.x, texture_coordinate.y)).r;
	yuv.y = texture(_t[1], vec2(texture_coordinate.x, texture_coordinate.y)).r;
	yuv.z = texture(_t[2], vec2(texture_coordinate.x, texture_coordinate.y)).r;

  vec3 rgb;

  yuv.x = 1.1643 * (yuv.x - 0.0625);
  yuv.y = yuv.y - 0.5;
  yuv.z = yuv.z - 0.5;

  rgb.x = yuv.x+ 1.5958* yuv.z;
  rgb.y = yuv.x - 0.391773 * yuv.y - 0.81290 * yuv.z;
  rgb.z = yuv.x + 2.017 * yuv.y;

	o_color = vec4(rgb.x, rgb.y, rgb.z, 1) * instance_color;
}