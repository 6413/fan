R"(
#version 130

out vec4 color;

uniform lowp sampler2D sampler_y;
uniform lowp sampler2D sampler_u;
uniform lowp sampler2D sampler_v;

in vec2 texture_coordinate;
in float render_depth;

void main() {

	vec3 yuv;

	yuv.x = texture2D(sampler_y, vec2(texture_coordinate.x, (1.0 - texture_coordinate.y))).r;
	yuv.y = texture2D(sampler_u, vec2(texture_coordinate.x, (1.0 - texture_coordinate.y))).r;
	yuv.z = texture2D(sampler_v, vec2(texture_coordinate.x, (1.0 - texture_coordinate.y))).r;

  vec3 rgb;

  yuv.x = 1.1643 * (yuv.x - 0.0625);
  yuv.y = yuv.y - 0.5;
  yuv.z = yuv.z - 0.5;

  rgb.x = yuv.x+ 1.5958* yuv.z;
  rgb.y = yuv.x - 0.391773 * yuv.y - 0.81290 * yuv.z;
  rgb.z = yuv.x + 2.017 * yuv.y;

	color = vec4(rgb.x, rgb.y, rgb.z, 1);
  gl_FragDepth = 0.999 - render_depth;
}
)"
