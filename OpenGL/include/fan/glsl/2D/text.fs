#version 330 core

in vec2 texture_coordinates;
out vec4 color;

uniform sampler2D texture_sampler;
uniform float original_font_size;

const float width = 0.5;
const float edge_multiplier = 0.08;

const float outline_width = 0.5;
const float outline_edge_multiplier = 0.5285;

in float font_size;
in vec4 outline_color;
in vec3 text_color;

void main() {

    float edge = edge_multiplier * original_font_size / font_size;

    float b = -0.00133;

    float outline_edge = max(0.05, b * font_size + outline_edge_multiplier) /* a */;

    float distance = 1.0 - texture(texture_sampler, texture_coordinates).a;
    float alpha = 1.0 - smoothstep(width, width + edge, distance);

    float outline_alpha = (1.0 - smoothstep(outline_width, outline_width + outline_edge, distance)) * outline_color.w;

    float letter_alpha = alpha + (1.0 - alpha) * outline_alpha;
    vec3 letter_color = mix(outline_color.xyz, text_color, alpha / letter_alpha);

    color = vec4(letter_color, letter_alpha);
} 