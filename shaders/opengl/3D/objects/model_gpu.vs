#version 410 core
layout (location = 0) in vec3 vertex;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 uv;
layout (location = 3) in vec4 bone_ids;
layout (location = 4) in vec4 bone_weights;
layout (location = 5) in vec3 vertex1;
layout (location = 6) in vec3 tangent;  
layout (location = 7) in vec3 bitangent;

out vec2 tex_coord;
out vec3 v_normal;
out vec3 v_pos;
out vec4 bw;

out vec3 c_tangent;
out vec3 c_bitangent;
//out vec3 c_bitangent ;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;
uniform mat4 m;

void main()
{
  vec3 v = vec3(vertex.x, vertex.y, vertex.z);
	gl_Position = projection * view * m * model * vec4(v, 1.0);
  tex_coord = uv;
  v_pos = vec3(model * vec4(v, 1.0));
	v_normal = mat3(transpose(inverse(model))) * normal;
	v_normal = normalize(v_normal);
  c_tangent = tangent;
  c_bitangent = bitangent;
  //v_bitangent = cross(normal, tangent);
}
