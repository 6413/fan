#version 460
#extension GL_EXT_ray_tracing          : require
#extension GL_EXT_nonuniform_qualifier : require

struct Payload {
    vec3  color;
    vec3  normal;
    vec2  uv;
    uint  material_id;
    float ao;
    int   depth;
};

layout(location = 0) rayPayloadInEXT Payload payload;
layout(location = 1) rayPayloadEXT bool shadowed;
hitAttributeEXT vec2 attribs;

struct MaterialInfo {
    int   albedo_texture_id;
    int   normal_texture_id;
    int   metallic_roughness_id;
    uint  padding;
    vec3  base_color;
    float pad1;
};

layout(binding = 0, set = 0) uniform accelerationStructureEXT topLevelAS;
layout(binding = 4, set = 0) uniform sampler2D textures[];
layout(binding = 5, set = 0) readonly buffer MaterialBuffer {
    MaterialInfo materials[];
};

struct Vertex {
    vec3 position; float pad0;
    vec3 normal;   float pad1;
    vec2 texcoord; vec2 pad2;
    vec3 color;    float pad3;
};

layout(binding = 6, set = 0) readonly buffer VertexBuffer {
    Vertex vertices[];
};

layout(binding = 7, set = 0) readonly buffer IndexBuffer {
    uint indices[];
};

layout(binding = 8, set = 0) uniform LightUBO {
    vec3 light_pos;
    float pad0;
    vec3 light_color;
    float intensity;
} light;

layout(binding = 9, set = 0) readonly buffer MaterialIndexBuffer {
    uint material_indices[];
};

layout(binding = 3, set = 0) uniform TimeUBO {
    float time;
} time_ubo;

layout(binding = 10, set = 0) uniform ExposureUBO {
    float exposure;
    float pad0;
    float pad1;
    float pad2;
} exposure_ubo;

float hash(vec3 p) {
    p = fract(p * 0.3183099 + vec3(0.1, 0.2, 0.3));
    p += dot(p, p.yzx + 19.19);
    return fract(p.x * p.y * p.z);
}

vec3 cosine_sample_hemisphere(vec3 N, float u1, float u2) {
    float r     = sqrt(u1);
    float theta = 6.2831853 * u2;

    float x = r * cos(theta);
    float y = r * sin(theta);
    float z = sqrt(max(0.0, 1.0 - u1));

    vec3 up = abs(N.z) < 0.999 ? vec3(0,0,1) : vec3(0,1,0);
    vec3 T  = normalize(cross(up, N));
    vec3 B  = cross(N, T);

    return normalize(T * x + B * y + N * z);
}

float luminance(vec3 c) {
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

float brightness(vec3 c) {
    return max(max(c.r, c.g), c.b);
}

void main() {

    uint prim_id     = gl_PrimitiveID;
    uint prim_offset = gl_InstanceCustomIndexEXT;
    uint global_prim = prim_offset + prim_id;

    uint         mat_id = material_indices[global_prim];
    MaterialInfo mat    = materials[mat_id];

    uint i0 = indices[global_prim * 3u + 0u];
    uint i1 = indices[global_prim * 3u + 1u];
    uint i2 = indices[global_prim * 3u + 2u];

    Vertex v0 = vertices[i0];
    Vertex v1 = vertices[i1];
    Vertex v2 = vertices[i2];

    vec3 bary = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);

    vec3 p0 = (gl_ObjectToWorldEXT * vec4(v0.position, 1.0)).xyz;
    vec3 p1 = (gl_ObjectToWorldEXT * vec4(v1.position, 1.0)).xyz;
    vec3 p2 = (gl_ObjectToWorldEXT * vec4(v2.position, 1.0)).xyz;

    vec3 e1 = p1 - p0;
    vec3 e2 = p2 - p0;

    vec3 Ng = normalize(cross(e1, e2));
    vec3 V  = normalize(-gl_WorldRayDirectionEXT);

    if (dot(Ng, V) < 0.0)
        Ng = -Ng;

    vec2 uv =
          v0.texcoord * bary.x +
          v1.texcoord * bary.y +
          v2.texcoord * bary.z;

    vec3 P = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;

    vec3 albedo = mat.base_color;
    if (mat.albedo_texture_id >= 0)
        albedo *= texture(textures[mat.albedo_texture_id], uv).rgb;

    vec3 N = Ng;

    if (mat.normal_texture_id >= 0) {
        vec3 n = texture(textures[mat.normal_texture_id], uv).xyz * 2.0 - 1.0;

        vec2 duv1 = v1.texcoord - v0.texcoord;
        vec2 duv2 = v2.texcoord - v0.texcoord;

        float det = duv1.x * duv2.y - duv1.y * duv2.x;
        float r = (abs(det) > 1e-8) ? 1.0 / det : 0.0;

        vec3 T = normalize((e1 * duv2.y - e2 * duv1.y) * r);
        vec3 B = normalize((e2 * duv1.x - e1 * duv2.x) * r);

        N = normalize(T * n.x + B * n.y + Ng * n.z);
    }

    // Ensure shading normal faces the view direction
    if (dot(N, V) < 0.0)
        N = -N;

    float metallic = 0.0;
    float roughness = 1.0;

    if (mat.metallic_roughness_id >= 0) {
        vec4 mr = texture(textures[mat.metallic_roughness_id], uv);
        metallic  = mr.b;
        roughness = mr.g;
    }

    metallic  = clamp(metallic, 0.0, 1.0);
    roughness = clamp(roughness, 0.04, 1.0);

    // Manual override for material ID 2: mirror-like
    if (mat_id == 2u) {
        metallic  = 1.0;
        roughness = 0.00;
    }

    float diffuse_strength = mix(1.0, 0.2, metallic);

    bool is_gi_ray = (payload.ao < 0.0);

    // GI ray shading
    if (is_gi_ray) {
        shadowed = false;

        vec3 Lp = light.light_pos;
        vec3 L  = normalize(Lp - P);
        float d = length(Lp - P);
        float NdotL = max(dot(N, L), 0.0);

        float sun_scale = 0.5;

        if (NdotL > 0.0) {
            traceRayEXT(
                topLevelAS,
                gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT,
                0xff,
                0, 0, 1,
                P + L * 0.01,
                0.001,
                L,
                d - 0.02,
                1
            );
        }

        vec3 ambient = albedo * 0.20;
        float shadow_mult = shadowed ? 0.25 : 1.0;

        vec3 diffuse = albedo * diffuse_strength *
                       light.light_color * light.intensity *
                       NdotL * shadow_mult * sun_scale;

        vec3 gi = ambient + diffuse;

        gi *= exposure_ubo.exposure;

        payload.color       = gi;
        payload.normal      = N;
        payload.uv          = uv;
        payload.material_id = mat_id;
        return;
    }

    // Secondary bounce shading
    if (payload.depth > 0) {
        float NdotV = max(dot(N, V), 0.0);
        float ao_view = pow(NdotV, 1.5);

        vec3 base = albedo * diffuse_strength * (0.2 + 0.8 * NdotV);
        base *= mix(0.85, 1.0, ao_view);

        base *= exposure_ubo.exposure;

        payload.color       = base;
        payload.normal      = N;
        payload.uv          = uv;
        payload.material_id = mat_id;
        return;
    }

    // Primary hit shading
    shadowed = false;

    vec3 Lp = light.light_pos;
    vec3 L  = normalize(Lp - P);
    float d = length(Lp - P);
    float NdotL = max(dot(N, L), 0.0);

    if (NdotL > 0.0) {
        traceRayEXT(
            topLevelAS,
            gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT,
            0xff,
            0, 0, 1,
            P + L * 0.01,
            0.001,
            L,
            d - 0.02,
            1
        );
    }

    float shadow_mult = shadowed ? 0.3 : 1.0;

    vec3 ambient = albedo * 0.08;

    vec3 diffuse =
        albedo * diffuse_strength *
        light.light_color * light.intensity *
        NdotL * shadow_mult;

    vec3 direct_lighting = ambient + diffuse;

    // One-bounce GI
    vec3 indirect_color = vec3(0.0);
    {
        Payload saved = payload;

        float fi = floor(time_ubo.time * 60.0);

        vec3 seed = vec3(
            float(gl_LaunchIDEXT.x) + 17.0 * fi,
            float(gl_LaunchIDEXT.y) + 31.0 * fi,
            13.0 * fi
        );

        float u1 = hash(seed);
        float u2 = hash(seed.yzx);

        vec3 gi_dir = cosine_sample_hemisphere(N, u1, u2);

        payload.color       = vec3(0.0);
        payload.normal      = vec3(0.0);
        payload.uv          = vec2(0.0);
        payload.material_id = 0u;
        payload.ao          = -1.0;
        payload.depth       = saved.depth + 1;

        traceRayEXT(
            topLevelAS,
            gl_RayFlagsOpaqueEXT,
            0xff,
            0, 0, 0,
            P + gi_dir * 0.01,
            0.001,
            gi_dir,
            10000.0,
            0
        );

        indirect_color = payload.color;
        payload = saved;
    }

    indirect_color = clamp(indirect_color, vec3(0.0), vec3(3.0));

    float direct_lum = luminance(direct_lighting);
    float gi_boost = smoothstep(0.0, 0.3, 1.0 - clamp(direct_lum, 0.0, 1.0));
    indirect_color *= mix(1.0, 2.0, gi_boost);

    // Mirror reflection for material 2
    vec3 reflection_color = vec3(0.0);
    if (mat_id == 2u) {
        vec3 R = reflect(-V, N);

        Payload saved = payload;

        payload.color       = vec3(0.0);
        payload.normal      = vec3(0.0);
        payload.uv          = vec2(0.0);
        payload.material_id = 0u;
        payload.ao          = 1.0;
        payload.depth       = saved.depth + 1;

        traceRayEXT(
            topLevelAS,
            gl_RayFlagsOpaqueEXT,
            0xff,
            0, 0, 0,
            P + R * 0.01,
            0.001,
            R,
            10000.0,
            0
        );

        reflection_color = payload.color;
				reflection_color *= light.intensity;

        payload = saved;
    }

    float NdotV = max(dot(N, V), 0.0);
    float ao_view = pow(NdotV, 1.5);
    float ao_factor = mix(0.8, 1.0, ao_view);

    vec3 final_color =
          direct_lighting
        + indirect_color
        + reflection_color;

    final_color *= ao_factor;
    final_color *= exposure_ubo.exposure;
    final_color = clamp(final_color, vec3(0.0), vec3(12.0));

    // Bloom
    float b = brightness(final_color);

    float bloom_thresh   = 1.0;
    float bloom_soft     = 0.5;
    float bloom_strength = 0.25;

    float bloom = smoothstep(bloom_thresh, bloom_thresh + bloom_soft, b);

    final_color += final_color * bloom * bloom_strength;

    payload.color       = final_color;
    payload.normal      = N;
    payload.uv          = uv;
    payload.material_id = mat_id;
}