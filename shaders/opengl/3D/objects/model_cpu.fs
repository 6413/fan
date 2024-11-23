#version 410 core

in vec2 tex_coord;
in vec3 v_normal;
in vec3 v_pos;
in vec4 bw;

in vec3 c_tangent;
in vec3 c_bitangent;
layout (location = 0) out vec4 color;

uniform sampler2D _t00; // aiTextureType_NONE
uniform sampler2D _t01; // aiTextureType_DIFFUSE
uniform sampler2D _t02; // aiTextureType_SPECULAR
uniform sampler2D _t03; // aiTextureType_AMBIENT
uniform sampler2D _t04; // aiTextureType_EMISSIVE
uniform sampler2D _t05; // aiTextureType_HEIGHT
uniform sampler2D _t06; // aiTextureType_NORMALS
uniform sampler2D _t07; // aiTextureType_SHININESS
uniform sampler2D _t08; // aiTextureType_OPACITY
uniform sampler2D _t09; // aiTextureType_DISPLACEMENT
uniform sampler2D _t10; // aiTextureType_LIGHTMAP
uniform sampler2D _t11; // aiTextureType_REFLECTION
uniform sampler2D _t12; // aiTextureType_BASE_COLOR
uniform sampler2D _t13; // aiTextureType_NORMAL_CAMERA
uniform sampler2D _t14; // aiTextureType_EMISSION_COLOR
uniform sampler2D _t15; // aiTextureType_METALNESS
uniform sampler2D _t16; // aiTextureType_DIFFUSE_ROUGHNESS
uniform sampler2D _t17; // aiTextureType_AMBIENT_OCCLUSION
uniform sampler2D _t18; // aiTextureType_SHEEN
uniform sampler2D _t19; // aiTextureType_CLEARCOAT
uniform sampler2D _t20; // aiTextureType_TRANSMISSION

void main()
{
	vec3 albedo = texture(_t12, tex_coord).rgb;
  color = vec4(albedo, 1);
}