
#version 130

out vec4 color;

uniform sampler2D _t00;
uniform sampler2D _t01;

in vec2 texture_coordinate;

void main() {

	vec3 yuv;

  vec3 yuv2r = vec3(1.164, 0.0, 1.596);
  vec3 yuv2g = vec3(1.164, -0.391, -0.813);
  vec3 yuv2b = vec3(1.164, 2.018, 0.0);

	yuv.x = texture2D(_t00, texture_coordinate).r  - 0.0625;
	yuv.y = texture2D(_t01, texture_coordinate).r - 0.5;
  yuv.z = texture2D(_t01, texture_coordinate).g - 0.5;

  vec3 rgb;

  rgb.x = dot(yuv, yuv2r);
  rgb.y = dot(yuv, yuv2g);
  rgb.z = dot(yuv, yuv2b);

	color = vec4(rgb.x, rgb.y, rgb.z, 1);
}
