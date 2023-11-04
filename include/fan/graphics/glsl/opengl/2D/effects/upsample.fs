
	#version 330

	layout (location = 0) out vec3 o_color;

	in vec2 texture_coordinate;

	in vec4 instance_color;

	uniform float filter_radius;

	uniform sampler2D _t00;

	void main() {
		// upsample
		// The filter kernel is applied with a radius, specified in texture
		// coordinates, so that the radius will vary across mip resolutions.
		float x = filter_radius;
		float y = filter_radius;

		// Take 9 samples around current texel:
		// a - b - c
		// d - e - f
		// g - h - i
		// === ('e' is the current texel) ===
		vec3 a = texture(_t00, vec2(texture_coordinate.x - x, texture_coordinate.y + y)).rgb;
		vec3 b = texture(_t00, vec2(texture_coordinate.x,     texture_coordinate.y + y)).rgb;
		vec3 c = texture(_t00, vec2(texture_coordinate.x + x, texture_coordinate.y + y)).rgb;

		vec3 d = texture(_t00, vec2(texture_coordinate.x - x, texture_coordinate.y)).rgb;
		vec3 e = texture(_t00, vec2(texture_coordinate.x,     texture_coordinate.y)).rgb;
		vec3 f = texture(_t00, vec2(texture_coordinate.x + x, texture_coordinate.y)).rgb;

		vec3 g = texture(_t00, vec2(texture_coordinate.x - x, texture_coordinate.y - y)).rgb;
		vec3 h = texture(_t00, vec2(texture_coordinate.x,     texture_coordinate.y - y)).rgb;
		vec3 i = texture(_t00, vec2(texture_coordinate.x + x, texture_coordinate.y - y)).rgb;

		// Apply weighted distribution, by using a 3x3 tent filter:
		//  1   | 1 2 1 |
		// -- * | 2 4 2 |
		// 16   | 1 2 1 |
		o_color = e*4.0;
		o_color += (b+d+f+h)*2.0;
		o_color += (a+c+g+i);
		o_color *= 1.0 / 16.0;
		//o_color = vec3(0, 0, 0);
	}
