#version 460
#extension GL_EXT_ray_tracing : require

layout(location = 0) rayPayloadInEXT vec3 debugColor;
layout(location = 1) rayPayloadEXT bool shadowed;

hitAttributeEXT vec2 attribs;

layout(binding = 0, set = 0) uniform accelerationStructureEXT topLevelAS;

layout(binding = 8, set = 0) uniform LightUBO {
	vec3 light_pos;
	float pad0;
	vec3 light_color;
	float intensity;
} light;

void main() {
	vec3 P = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;

	vec3 Lp = light.light_pos;
	vec3 L  = normalize(Lp - P);
	float d = length(Lp - P);

	shadowed = true;
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

	debugColor = shadowed ? vec3(0.0) : vec3(1.0);
}
