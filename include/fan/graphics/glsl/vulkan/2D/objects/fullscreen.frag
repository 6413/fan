#version 450

layout(input_attachment_index=0, binding=4) uniform subpassInput attachment0;
layout(input_attachment_index=1, binding=5) uniform subpassInput attachment1;

layout(location = 0) out vec4 ocolor;
 
// epsilon number
const float EPSILON = 0.00001f;

// calculate floating point numbers equality accurately
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

	vec4 accumulation = subpassLoad(attachment0);
	
	if (isinf(max3(abs(accumulation.rgb))))
		accumulation.rgb = vec3(accumulation.a);
	
	float revealage = subpassLoad(attachment1).r;

	vec3 average_color = accumulation.rgb / max(accumulation.a, EPSILON);

	ocolor = mix(vec4(average_color, 1.0 - revealage), accumulation, revealage);
}