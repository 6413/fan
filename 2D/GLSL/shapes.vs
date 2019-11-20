#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;

out vec3 FragPos; 
out vec3 Normal;

uniform mat4 view;
uniform mat4 projection;


void main() {
	 Normal = aNormal;  
	 FragPos = vec3(vec4(aPos, 1.0));
	 gl_Position = projection * view * vec4(aPos.x, aPos.y, aPos.z, 1.0);
}