#version 450

//layout(location = 2) out vec4 out_color;
layout(location = 0) out vec4 ocolor;
layout(location = 1) out float reveal;

struct travel_data_t {
	vec4 color;
	float depth;
};

layout(location = 0) in travel_data_t data;

void main() {
  vec4 color = data.color;
	float weight = clamp(pow(min(1.0, color.a * 10.0) + 0.01, 3.0) * 1e8 * 
											 pow(1.0 - gl_FragCoord.z * 0.9, 3.0), 1e-2, 3e3);
	reveal = color.a;
	//out_color = color;
	ocolor = vec4(color.rgb * color.a, color.a) * weight;
}