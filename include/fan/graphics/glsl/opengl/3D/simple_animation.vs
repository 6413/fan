R"(

#version 440 core
layout (location = 0) in vec3 position; 
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 uv;

out vec2 tex_cord;
out vec3 v_normal;
out vec3 v_pos;
out vec4 bw;

uniform mat4 models[50];
uniform mat4 projection;
uniform mat4 view;

void main()
{
	int index = 0;

	vec4 pos = vec4(position, 1.0);
	gl_Position = (projection * view) * models[index] * pos;
	v_pos = vec3(models[index] * pos);
	tex_cord = uv;
	v_normal = mat3(transpose(inverse(models[index]))) * normal;
	v_normal = normalize(v_normal);
}
)"