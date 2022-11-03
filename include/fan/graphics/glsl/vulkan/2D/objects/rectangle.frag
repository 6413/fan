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
	color.rgb *= color.a;

  // Insert your favorite weighting function here. The color-based factor
  // avoids color pollution from the edges of wispy clouds. The z-based
  // factor gives precedence to nearer surfaces.

  // The depth functions in the paper want a camera-space depth of 0.1 < z < 500,
  // but the scene at the moment uses a range of about 0.01 to 50, so multiply
  // by 10 to get an adjusted depth:
  const float depthZ = -data.depth * 10.0f;

  const float distWeight = clamp(0.03 / (1e-5 + pow(depthZ / 200, 4.0)), 1e-2, 3e3);

  float alphaWeight = min(1.0, max(max(color.r, color.g), max(color.b, color.a)) * 40.0 + 0.01);
  alphaWeight *= alphaWeight;

  const float weight = alphaWeight * distWeight;

  ocolor = color * weight;

  reveal = color.a;
}