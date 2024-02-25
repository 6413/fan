#version 330

layout (location = 1) out vec4 o_attachment1;

in vec4 instance_color;
in vec3 instance_position;
in vec2 instance_size;
in vec3 frag_position;

#define light_count 50

uniform vec2 un_positions[light_count];

float raySphereIntersect(vec3 r0, vec3 rd, vec3 s0, float sr) {
    float a = dot(rd, rd);
    vec3 s0_r0 = r0 - s0;
    float b = 2.0 * dot(rd, s0_r0);
    float c = dot(s0_r0, s0_r0) - (sr * sr);
    if (b * b - 4.0 * a * c < 0.0) {
        return -1.0;
    }
    return (-b - sqrt((b * b) - 4.0 * a * c)) / (2.0 * a);
}

bool isSameDirection(vec2 src, vec2 dst) {
    vec2 direction = normalize(dst);
    vec2 ray = normalize(src);

    float dotProduct = dot(direction, ray);

    return dotProduct > 0.0;
}

void main() {
    vec4 lightColor = vec4(0.0, 0.0, 0.0, 1.0); // Default color

    // Sphere 1
    vec3 rayDirection1 = normalize(instance_position - frag_position);
    rayDirection1.z = 0;

    for (int i = 0; i < light_count; ++i) {
      vec3 sphereCenter1 = vec3(un_positions[i], 0);
      float sphereRadius1 = 25.0;

      float d1 = raySphereIntersect(frag_position, rayDirection1, sphereCenter1, sphereRadius1);
      if (d1 > -1 && d1 < instance_size.x && isSameDirection(-rayDirection1.xy, vec3(sphereCenter1 - instance_position).xy)) {
        lightColor = vec4(vec3(0), 1.0); // Shadowed by the sphere
        o_attachment1 = lightColor;
        return;
      }
    }

    // Calculate light color if not shadowed by any sphere
    float distance = length(frag_position - instance_position);
    float radius = instance_size.x;
    float smooth_edge = radius;
    float intensity = 1.0 - smoothstep(radius / 3 - smooth_edge, radius, distance);
    lightColor = instance_color * intensity;

    o_attachment1 = lightColor;
}
