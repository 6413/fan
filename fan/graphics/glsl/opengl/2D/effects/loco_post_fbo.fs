#version 330 core

in vec2 texture_coordinate;
layout (location = 0) out vec4 o_attachment0;

uniform sampler2D _t00;
uniform float m_time;

float iTime = m_time;

void main() {
    // Sample from the texture using the transformed coordinates
    vec4 color = texture(_t00, texture_coordinate);
    // Output the final color
    o_attachment0 = vec4(color.rgb, 1.0);
}
