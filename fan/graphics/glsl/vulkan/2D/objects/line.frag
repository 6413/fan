R"(
#version 430

layout(location = 0) out vec4 ocolor;
layout(location = 1) out vec4 rcolor;

struct travel_data_t {
	vec4 color;
};

layout(location = 0) in travel_data_t data;

void main() {
  rcolor = vec4(0);
	ocolor = data.color;
}
)"