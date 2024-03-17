#version 330

layout (location = 1) out vec4 o_attachment1;

in vec4 instance_color;
in vec3 instance_position;
in vec2 instance_size;
in vec3 frag_position;

void main() {
    vec4 lightColor = vec4(0.0, 0.0, 0.0, 1.0); // Default color

    // Calculate light color if not shadowed by any sphere
    float distance = length(frag_position - instance_position);
    float radius = instance_size.x;
    float smooth_edge = radius;
    float intensity = 1.0 - smoothstep(radius / 3 - smooth_edge, radius, distance);
    lightColor = instance_color * intensity;

    o_attachment1 = lightColor;
}
