#version 430

//#define wboit

#if !defined(wboit)
	layout(location = 0) out vec4 ocolor;
#else
	layout(location = 0) out vec4 ocolor;
	layout(location = 1) out float reveal;
#endif

struct travel_data_t {
	vec4 color;
	float depth;
};

layout(location = 0) in travel_data_t data;

void main() {
	#if !defined(wboit)
		ocolor = data.color;
	#else
		vec4 color = data.color;
		if (distance(gl_FragCoord.xyz, vec3(400, 400, 0)) < 100) {
			color.a = 1.0;
		}
		color.rgb *= color.a;  // Premultiply it

		// Insert your favorite weighting function here. The color-based factor
		// avoids color pollution from the edges of wispy clouds. The z-based
		// factor gives precedence to nearer surfaces.

		// The depth functions in the paper want a camera-space depth of 0.1 < z < 500,
		// but the scene at the moment uses a range of about 0.01 to 50, so multiply
		// by 10 to get an adjusted depth:
		const float depthZ = -data.depth * 10;

		const float distWeight = clamp(3.0 / (1e-5 + pow(depthZ / 200, 4.0)), 1e-2, 3e3);

		float alphaWeight = min(1.0, max(max(color.r, color.g), max(color.b, color.a)) * 40.0 + 0.01);
		alphaWeight *= alphaWeight;

		const float weight = alphaWeight * distWeight;

		// GL Blend function: GL_ONE, GL_ONE
		ocolor = color * weight;

		// GL blend function: GL_ZERO, GL_ONE_MINUS_SRC_ALPHA
		reveal = color.a;
	#endif
}