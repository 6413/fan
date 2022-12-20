#version 420

//layout(input_attachment_index=0, binding=4) uniform subpassInput attachment0;
//layout(input_attachment_index=1, binding=5) uniform subpassInput attachment1;

//layout(location = 0) out vec4 o_color;
layout(location = 0) out vec4 r_color;
//layout(location = 2) out vec4 o_color;

void main() {
	r_color = vec4(1, 0, 0, 1);
	//vec4 accum = subpassLoad(attachment0);
	//float reveal = subpassLoad(attachment1).r;
	//
	//o_color = vec4(accum.rgb / max(accum.a, 1e-5), reveal);
}