R"(
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
in vec4 outline_color;
in float outline_size;

out vec4 color;

void main() {

    float smoothing = 1.0 / (font_size / 8);
    float outline_width = outline_size / font_size;
    float outer_edge_center = 0.5 - outline_width;

    float distance = TEXTURE2D(texture_sampler, texture_coordinate).r;
    float alpha = smoothstep(outer_edge_center - smoothing, outer_edge_center + smoothing, distance);
    float border = smoothstep(0.5 - smoothing, 0.5 + smoothing, distance);

    if (outline_color.a == 0) {
        color = vec4(vec3(text_color.rgb), alpha);
    }
    else {
        color = vec4(mix(outline_color.rgb, text_color.rgb, border), alpha);
    }
}
)"