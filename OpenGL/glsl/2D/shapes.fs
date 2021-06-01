#version 120

#if __VERSION__ < 130
#define TEXTURE2D texture2D
#else
#define TEXTURE2D texture
#endif


//precision highp float;

varying vec2 texture_coordinate;

varying vec4 color;

uniform int enable_texture;
uniform sampler2D texture_sampler;

void main()
{
	if (enable_texture > 0) { // ei toimi
		vec2 flipped_y = vec2(texture_coordinate.x, 1.0 - texture_coordinate.y);

		gl_FragColor = TEXTURE2D(texture_sampler, flipped_y);
	}
	else {
		gl_FragColor = color;
	}

} 