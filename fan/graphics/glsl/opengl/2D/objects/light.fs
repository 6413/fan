#version 330

layout (location = 1) out vec4 o_attachment1;
layout (location = 3) out vec4 o_attachment3;

in vec4 instance_color;
in vec3 instance_position;
in vec2 instance_size;
in vec3 frag_position;

flat in uint fs_flags;

void main() {
    vec4 lightColor = vec4(0.0, 0.0, 0.0, 1.0);

    float distance = length(frag_position - instance_position);
    float radius = instance_size.x;
    float smooth_edge = radius;
    float intensity = 1.0 - smoothstep(radius / 3 - smooth_edge, radius, distance);
    lightColor = instance_color * intensity;

    o_attachment1 = lightColor;

    if ((fs_flags & 0x2u) == 0x2u) {
      o_attachment3 = lightColor;
    }
    else {
      o_attachment3 = vec4(0, 0, 0, 0);
    }
}
