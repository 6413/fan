#version 330 core

in vec2 texture_coordinates;
out vec4 color;

uniform sampler2D texture_sampler;
uniform float original_font_size;

in float font_size;
in vec4 outline_color;
in vec3 text_color;

void main() {

    float smoothing = 1.0 / (font_size / 2);

    float distance = texture(texture_sampler, texture_coordinates).a;
    float alpha = smoothstep(0.5 - smoothing, 0.5 + smoothing, distance);

    color = vec4(text_color, alpha);
} 