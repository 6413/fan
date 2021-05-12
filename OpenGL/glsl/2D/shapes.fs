#version 120

#if __VERSION__ < 130
#define TEXTURE2D texture2D
#else
#define TEXTURE2D texture
#endif


//precision highp float;

varying vec2 texture_coordinate;

varying vec4 color;

uniform bool enable_texture;
uniform sampler2D texture_sampler;

void main()
{
	if (false) {
		gl_FragColor = TEXTURE2D(texture_sampler, texture_coordinate);
	}
	else {
		gl_FragColor = color;
	}

} 