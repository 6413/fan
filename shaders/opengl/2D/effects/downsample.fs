#version 330 core
layout (location = 0) out vec3 o_color;
in vec2 texture_coordinate;

uniform sampler2D _t00;
uniform vec2 resolution;
uniform int mipLevel = 1;
uniform float threshold = 1.0;
uniform float knee = 0.1;

// Helper function to isolate the bright pixels
vec3 apply_threshold(vec3 col) {
    float brightness = max(col.r, max(col.g, col.b));
    float rq = clamp(brightness - threshold + knee, 0.0, 2.0 * knee);
    rq = (rq * rq) / (4.0 * knee + 0.0001);
    float mask = max(rq, brightness - threshold) / max(brightness, 0.0001);
    return col * mask;
}

void main() {
    vec2 srcTexelSize = 1.0 / resolution;
    float x = srcTexelSize.x;
    float y = srcTexelSize.y;

   if (mipLevel == 0) {
        vec3 center = texture(_t00, texture_coordinate).rgb;
        vec3 a = texture(_t00, vec2(texture_coordinate.x - x, texture_coordinate.y + y)).rgb;
        vec3 b = texture(_t00, vec2(texture_coordinate.x + x, texture_coordinate.y + y)).rgb;
        vec3 c = texture(_t00, vec2(texture_coordinate.x - x, texture_coordinate.y - y)).rgb;
        vec3 d = texture(_t00, vec2(texture_coordinate.x + x, texture_coordinate.y - y)).rgb;

        center = apply_threshold(center);
        a = apply_threshold(a);
        b = apply_threshold(b);
        c = apply_threshold(c);
        d = apply_threshold(d);

        o_color = max(center, max(max(a, b), max(c, d)));
    }
    else {
        vec3 a = texture(_t00, vec2(texture_coordinate.x - 2.0*x, texture_coordinate.y + 2.0*y)).rgb;
        vec3 b = texture(_t00, vec2(texture_coordinate.x,         texture_coordinate.y + 2.0*y)).rgb;
        vec3 c = texture(_t00, vec2(texture_coordinate.x + 2.0*x, texture_coordinate.y + 2.0*y)).rgb;

        vec3 d = texture(_t00, vec2(texture_coordinate.x - 2.0*x, texture_coordinate.y)).rgb;
        vec3 e = texture(_t00, vec2(texture_coordinate.x,         texture_coordinate.y)).rgb;
        vec3 f = texture(_t00, vec2(texture_coordinate.x + 2.0*x, texture_coordinate.y)).rgb;

        vec3 g = texture(_t00, vec2(texture_coordinate.x - 2.0*x, texture_coordinate.y - 2.0*y)).rgb;
        vec3 h = texture(_t00, vec2(texture_coordinate.x,         texture_coordinate.y - 2.0*y)).rgb;
        vec3 i = texture(_t00, vec2(texture_coordinate.x + 2.0*x, texture_coordinate.y - 2.0*y)).rgb;

        vec3 j = texture(_t00, vec2(texture_coordinate.x - x, texture_coordinate.y + y)).rgb;
        vec3 k = texture(_t00, vec2(texture_coordinate.x + x, texture_coordinate.y + y)).rgb;
        vec3 l = texture(_t00, vec2(texture_coordinate.x - x, texture_coordinate.y - y)).rgb;
        vec3 m = texture(_t00, vec2(texture_coordinate.x + x, texture_coordinate.y - y)).rgb;

        vec3 col = e*0.125;
        col += (a+c+g+i)*0.03125;
        col += (b+d+f+h)*0.0625;
        col += (j+k+l+m)*0.125;
        
        o_color = max(col, 0.0001);
    }
}