#version 430 core

in vec2 texture_coordinate;
out vec4 shape_color;

uniform sampler2D texture_sampler;

in vec2 hori_blur_texture_coordinates[11];
in vec2 vert_blur_texture_coordinates[11];

void main() {
    //shape_color = texture(texture_sampler, texture_coordinate);

    vec4 vert_color, hori_color;

    vert_color += texture(texture_sampler, vert_blur_texture_coordinates[0]) * 0.0093;
    vert_color += texture(texture_sampler, vert_blur_texture_coordinates[1]) * 0.028002;
    vert_color += texture(texture_sampler, vert_blur_texture_coordinates[2]) * 0.065984;
    vert_color += texture(texture_sampler, vert_blur_texture_coordinates[3]) * 0.121703;
    vert_color += texture(texture_sampler, vert_blur_texture_coordinates[4]) * 0.175713;
    vert_color += texture(texture_sampler, vert_blur_texture_coordinates[5]) * 0.198596;
    vert_color += texture(texture_sampler, vert_blur_texture_coordinates[6]) * 0.175713;
    vert_color += texture(texture_sampler, vert_blur_texture_coordinates[7]) * 0.121703;
    vert_color += texture(texture_sampler, vert_blur_texture_coordinates[8]) * 0.065984;
    vert_color += texture(texture_sampler, vert_blur_texture_coordinates[9]) * 0.028002;
    vert_color += texture(texture_sampler, vert_blur_texture_coordinates[10]) * 0.0093;

    hori_color += texture(texture_sampler, hori_blur_texture_coordinates[0]) * 0.0093  ;
    hori_color += texture(texture_sampler, hori_blur_texture_coordinates[1]) * 0.028002;
    hori_color += texture(texture_sampler, hori_blur_texture_coordinates[2]) * 0.065984;
    hori_color += texture(texture_sampler, hori_blur_texture_coordinates[3]) * 0.121703;
    hori_color += texture(texture_sampler, hori_blur_texture_coordinates[4]) * 0.175713;
    hori_color += texture(texture_sampler, hori_blur_texture_coordinates[5]) * 0.198596;
    hori_color += texture(texture_sampler, hori_blur_texture_coordinates[6]) * 0.175713;
    hori_color += texture(texture_sampler, hori_blur_texture_coordinates[7]) * 0.121703;
    hori_color += texture(texture_sampler, hori_blur_texture_coordinates[8]) * 0.065984;
    hori_color += texture(texture_sampler, hori_blur_texture_coordinates[9]) * 0.028002;
    hori_color += texture(texture_sampler, hori_blur_texture_coordinates[10]) * 0.0093;


    shape_color = mix(vert_color, hori_color, 0.5);


 //shape_color *= vert_color;

} 