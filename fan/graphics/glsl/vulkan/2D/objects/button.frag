R"(
#version 450

layout(location = 0) out vec4 color;
layout(location = 1) out vec4 rcolor;

struct in_data_t {
	vec4 color;
	vec4 outline_color;
	vec2 tc;
	float outline_size;
	float aspect_ratio;
};

layout(location = 0) in in_data_t in_data;

void main() {
	vec2 p = abs(in_data.tc);
	vec2 border_size = vec2(1.0) - in_data.outline_size * vec2(in_data.aspect_ratio, 1);
  rcolor = vec4(0);
  if (p.x > border_size.x) {
		color = in_data.outline_color;
	}
	else if (p.y > border_size.y) {
		color = in_data.outline_color;
	}
	else {
		color = in_data.color;
	}
}
)"