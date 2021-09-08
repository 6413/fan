#version 450

layout(location = 0) out vec4 color;
layout(location = 0) in vec2 texture_coordinate;
layout(binding = 1) uniform sampler2D texture_sampler;

layout(location = 1) in float font_size;

layout(location = 2) in vec4 text_color;

void main() {

    float smoothing = 1.0 / (font_size / 2);

    float distance = texture(texture_sampler, vec2(texture_coordinate.x, texture_coordinate.y)).a;
    float alpha = smoothstep(0.5 - smoothing, 0.5 + smoothing, distance);
   
    color = vec4(vec3(text_color.rgb), alpha);
} 