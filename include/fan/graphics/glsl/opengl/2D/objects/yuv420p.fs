R"(
#version 130

out vec4 color;

uniform sampler2D _t00;
uniform sampler2D _t01;
uniform sampler2D _t02;

in vec4 instance_color;
in vec2 texture_coordinate;

void main() {

	vec3 yuv;

	//yuv.x = texture2D(_t00, vec2(texture_coordinate.x, texture_coordinate.y)).r;
	//yuv.y = texture2D(_t01, vec2(texture_coordinate.x, texture_coordinate.y)).r;
	//yuv.z = texture2D(_t02, vec2(texture_coordinate.x, texture_coordinate.y)).r;
  //
  vec3 rgb;
  //
  //yuv.x = 1.1643 * (yuv.x - 0.0625);
  //yuv.y = yuv.y - 0.5;
  //yuv.z = yuv.z - 0.5;
  //
  //rgb.x = yuv.x+ 1.5958* yuv.z;
  //rgb.y = yuv.x - 0.391773 * yuv.y - 0.81290 * yuv.z;
  //rgb.z = yuv.x + 2.017 * yuv.y;
  //
	//color = vec4(rgb.x, rgb.y, rgb.z, 1) * instance_color;

    vec3 yuv2r = vec3(1.164, 0.0, 1.596);
    vec3 yuv2g = vec3(1.164, -0.391, -0.813);
    vec3 yuv2b = vec3(1.164, 2.018, 0.0);

    yuv.x = texture2D(_t00, vec2(texture_coordinate.x, texture_coordinate.y)).r- 0.0625;
    yuv.y = texture2D(_t01, vec2(texture_coordinate.x, texture_coordinate.y)).r - 0.5;
    yuv.z = texture2D(_t02, vec2(texture_coordinate.x, texture_coordinate.y)).r - 0.5;

    rgb.x = dot(yuv, yuv2r);
    rgb.y = dot(yuv, yuv2g);
    rgb.z = dot(yuv, yuv2b);
  color = vec4(texture2D(_t01, vec2(texture_coordinate.x, texture_coordinate.y)).r, 0, 0, 1) * instance_color;
}
)"