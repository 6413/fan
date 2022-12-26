R"(
#version 430

layout(location = 0) out vec4 ocolor;

struct travel_data_t {
	vec4 color;
};

layout(location = 0) in travel_data_t data;

void main() {
	ocolor = data.color;
}
)"