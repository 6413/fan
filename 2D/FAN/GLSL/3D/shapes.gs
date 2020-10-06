#version 430 core
layout (points) in;
layout (triangle_strip, max_vertices = 512) out;

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
    vec4 start_pos = gl_in[0].gl_Position + vec4(-0.5, -0.5, 0.0, 0.0);
    vec4 bottom_vertex = start_pos;

    fColor = vec3(1, 1,1);
    mat4 MPV = get_projection_view(0);

   for (int i = 37; i >= 0; i--) {
        gl_Position = MVP * vec4( 0, 0 , 0, 1 );
        EmitVertex(); // epicenter

        gl_Position = MVP * vec4( data[i], 0, 1 );
        EmitVertex();

        gl_Position = MVP * vec4( data[i-1], 0, 1 );
        EmitVertex();

        // Fan and strip DNA just won't splice
        EndPrimitive ();
     }
}