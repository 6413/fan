#version 430 core
layout (points) in;
layout (triangle_strip, max_vertices = 3) out;

struct custom_vec3 {
  float x[3];
};

layout(std430, binding = 0) buffer texture_coordinate_layout 
{
    custom_vec3 ivertices[];
};

out vec3 fColor;

uniform mat4 projection;
uniform mat4 view;

vec4 get_projection_view(vec4 position) {
    return projection * view * position;
}

void main() {    
    vec4 position = gl_in[0].gl_Position;
    for (int j = 0; j < ivertices.length() / 3; j++) {
        for (int i = 0; i < 3; i++) {
            gl_Position = get_projection_view(vec4(position.x + ivertices[i * j].x[0], position.y + ivertices[i * j].x[1], position.z + ivertices[i * j].x[2], position.w)); 
            fColor = vec3(1.0, 1.0, 1.0);
            EmitVertex();
        }

        EndPrimitive();
    }

}