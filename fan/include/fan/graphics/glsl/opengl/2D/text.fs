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

const float border_width = 0.4;
const float border_edge = 0.5;

void main() {


    if (outline_color.a != 0) {
        float width = outline_size;
        float smoothing = 1.0 / (font_size / 12);

        float distance = 1.0 - TEXTURE2D(texture_sampler, texture_coordinate).r;
        float alpha = 1.0 - smoothstep(width, width + smoothing, distance);

        float distance2 = 1.0 - TEXTURE2D(texture_sampler, texture_coordinate).r;
        float outline_alpha = 1.0 - smoothstep(border_width, border_width + border_edge, distance2);

        float final_alpha = alpha + (1.0 - alpha) * outline_alpha;
        vec3 final_color = mix(outline_color.rgb, text_color.rgb, alpha / final_alpha);

        color = vec4(final_color, final_alpha);
    }
    else {
        float width = 0.5;
        float smoothing = 1.0 / (font_size / 8);
        float distance = 1.0 - TEXTURE2D(texture_sampler, texture_coordinate).r;
        float alpha = 1.0 - smoothstep(width, width + smoothing, distance);

        color = vec4(text_color.rgb, alpha);
    }
}
)"