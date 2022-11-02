#version 450

layout(input_attachment_index=0, binding=4) uniform subpassInput attachment0;
layout(input_attachment_index=1, binding=5) uniform subpassInput attachment1;

layout(location = 0) out vec4 ocolor;
 
// epsilon number
const float EPSILON = 0.00001f;

// caluclate floating point numbers equality accurately
bool isApproximatelyEqual(float a, float b)
{
	return abs(a - b) <= (abs(a) < abs(b) ? abs(b) : abs(a)) * EPSILON;
}

// get the max value between three values
float max3(vec3 v) 
{
	return max(max(v.x, v.y), v.z);
}

void main() {
	vec4 accum = subpassLoad(attachment0);
	float reveal = subpassLoad(attachment1).r;

	ocolor = vec4(accum.rgb / max(accum.a, 1e-5), reveal);
}