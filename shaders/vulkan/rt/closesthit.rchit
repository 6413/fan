#version 460
#extension GL_EXT_ray_tracing : require

struct Payload {
    vec3 color;
};

layout(location = 0) rayPayloadInEXT Payload payload;

hitAttributeEXT vec2 attribs;

layout(binding = 3, set = 0) uniform TimeUBO {
    float time;
} ubo;

vec3 hash3(float n) {
    return fract(sin(vec3(n, n + 1.0, n + 2.0)) * 43758.5453123);
}

void main() {

    int pid = gl_InstanceCustomIndexEXT;
    float seed = float(pid);

    vec3 rnd = hash3(seed);

    float speed = 0.2 + hash3(seed + 123.45).x * 1.5;

    float phase = hash3(seed + 999.0).y * 6.2831853; 

    float t = ubo.time * speed + phase;

    vec3 centerOffset = vec3(
        sin(t + rnd.x * 3.0),
        cos(t + rnd.y * 5.0),
        sin(t * 0.7 + rnd.z * 2.0)
    );

    vec3 hitPos = gl_WorldRayOriginEXT +
                  gl_HitTEXT * gl_WorldRayDirectionEXT;

    vec3 center = centerOffset;

    vec3 N = normalize(hitPos - center);

    payload.color = N * 0.5 + 0.5;
}