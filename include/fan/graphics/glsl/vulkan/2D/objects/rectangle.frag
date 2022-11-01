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
	ocolor = vec4(0, 0, 1, 1);
	return;
  vec4 color = data.color;
	color.rgb *= color.a;
	const float depthZ = -data.depth * 10.0f;

  const float distWeight = clamp(0.03 / (1e-5 + pow(depthZ / 200, 4.0)), 1e-2, 3e3);

  float alphaWeight = min(1.0, max(max(color.r, color.g), max(color.b, color.a)) * 40.0 + 0.01);
  alphaWeight *= alphaWeight;

  const float weight = alphaWeight * distWeight;
	
	reveal = color.a;
	//out_color = color;
	ocolor = vec4(0, 0, 1, 1);
}