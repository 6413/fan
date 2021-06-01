#version 120

#if __VERSION__ < 130
#define TEXTURE2D texture2D
#else
#define TEXTURE2D texture
#endif

uniform sampler2D texture_sampler;
uniform float original_font_size;

varying vec2 texture_coordinates;
varying float font_size;
varying vec3 text_color;

void main() {

    float smoothing = 1.0 / (font_size / 2);

    float distance = TEXTURE2D(texture_sampler, texture_coordinates).a;
    float alpha = smoothstep(0.5 - smoothing, 0.5 + smoothing, distance);
   
    gl_FragColor = vec4(text_color, alpha);
} 