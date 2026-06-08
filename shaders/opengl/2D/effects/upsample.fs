#version 330 core
layout (location = 0) out vec3 o_color;
in vec2 texture_coordinate;

uniform vec2 filter_radius;
uniform sampler2D _t00;

void main() {
    float x = filter_radius.x;
    float y = filter_radius.y;

    vec3 a = texture(_t00, vec2(texture_coordinate.x - x, texture_coordinate.y + y)).rgb;
    vec3 b = texture(_t00, vec2(texture_coordinate.x,     texture_coordinate.y + y)).rgb;
    vec3 c = texture(_t00, vec2(texture_coordinate.x + x, texture_coordinate.y + y)).rgb;

    vec3 d = texture(_t00, vec2(texture_coordinate.x - x, texture_coordinate.y)).rgb;
    vec3 e = texture(_t00, vec2(texture_coordinate.x,     texture_coordinate.y)).rgb;
    vec3 f = texture(_t00, vec2(texture_coordinate.x + x, texture_coordinate.y)).rgb;

    vec3 g = texture(_t00, vec2(texture_coordinate.x - x, texture_coordinate.y - y)).rgb;
    vec3 h = texture(_t00, vec2(texture_coordinate.x,     texture_coordinate.y - y)).rgb;
    vec3 i = texture(_t00, vec2(texture_coordinate.x + x, texture_coordinate.y - y)).rgb;

    o_color = e * 4.0;
    o_color += (b + d + f + h) * 2.0;
    o_color += (a + c + g + i);
    o_color *= 1.0 / 16.0;
}