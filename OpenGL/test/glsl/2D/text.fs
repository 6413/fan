#version 130

#if __VERSION__ < 130
#define TEXTURE2D texture2D
#else
#define TEXTURE2D texture
#endif

uniform sampler2D texture_sampler;

in vec2 texture_coordinate;
in float font_size;
in vec4 text_color;

out vec4 color;

void main() {

    float smoothing = 1.0 / (font_size / 2);

    float distance = TEXTURE2D(texture_sampler, texture_coordinate).a;
    float alpha = smoothstep(0.5 - smoothing, 0.5 + smoothing, distance);
   
    color = vec4(vec3(text_color.rgb), alpha);
} 