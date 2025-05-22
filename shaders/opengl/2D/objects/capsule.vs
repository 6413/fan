#version 330
layout (location = 0) in vec3 in_position;
layout (location = 1) in vec2 in_center0;
layout (location = 2) in vec2 in_center1;
layout (location = 3) in float in_radius;
layout (location = 4) in vec2 in_rotation_point;
layout (location = 5) in vec4 in_color;
layout (location = 6) in vec3 in_angle;
layout (location = 7) in uint in_flags;
layout (location = 8) in vec4 in_outline_color;

out vec4 instance_color;
out vec4 instance_outline_color;
out vec3 instance_position;
out float instance_radius;
out vec2 instance_center0;
out vec2 instance_center1;
out vec3 frag_position;
out vec2 texture_coordinate;
flat out uint flags;

uniform mat4 view;
uniform mat4 projection;

vec2 rectangle_vertices[] = vec2[](
    vec2(-1.0, -1.0),
    vec2(1.0, -1.0),
    vec2(1.0, 1.0),

    vec2(1.0, 1.0),
    vec2(-1.0, 1.0),
    vec2(-1.0, -1.0)
);

vec2 tc[] = vec2[](
    vec2(0, 0), // top left
    vec2(1, 0), // top right
    vec2(1, 1), // bottom right
    vec2(1, 1), // bottom right
    vec2(0, 1), // bottom left
    vec2(0, 0) // top left
);

mat4 rotate(mat4 m, vec3 angles) {
    float cx = cos(angles.x);
    float sx = sin(angles.x);
    float cy = cos(angles.y);
    float sy = sin(angles.y);
    float cz = cos(angles.z);
    float sz = sin(angles.z);

    mat4 rotationX = mat4(1.0, 0.0, 0.0, 0.0,
                          0.0, cx, -sx, 0.0,
                          0.0, sx, cx, 0.0,
                          0.0, 0.0, 0.0, 1.0);

    mat4 rotationY = mat4(cy, 0.0, sy, 0.0,
                          0.0, 1.0, 0.0, 0.0,
                          -sy, 0.0, cy, 0.0,
                          0.0, 0.0, 0.0, 1.0);

    mat4 rotationZ = mat4(cz, -sz, 0.0, 0.0,
                          sz, cz, 0.0, 0.0,
                          0.0, 0.0, 1.0, 0.0,
                          0.0, 0.0, 0.0, 1.0);

    mat4 matrix = rotationX * rotationY * rotationZ * m;
    return matrix;
}

mat2 createRotationMatrix(float angle) {
    return mat2(
        cos(angle), -sin(angle),
        sin(angle), cos(angle)
    );
}
void main() {
    vec2 rp = rectangle_vertices[gl_VertexID % 6];
    texture_coordinate = tc[gl_VertexID % 6];
    
    // Pass through instance data
    instance_color = in_color;
    instance_outline_color = in_outline_color;
    instance_radius = in_radius;
    mat4 rr = mat4(1);
    rr = rotate(rr, -in_angle);
    instance_center0 = (rr * vec4(in_center0, 0, 1)).xy + in_position.xy;
    instance_center1 = (rr * vec4(in_center1, 0, 1)).xy + in_position.xy;
    flags = in_flags;
    
    vec2 capsule_dir = in_center1 - in_center0;
    float capsule_length = length(capsule_dir);
    float angle = atan(capsule_dir.y, capsule_dir.x) + in_angle.z;

    float box_margin = in_radius * 2.0;
    vec2 box_size = vec2(
        capsule_length + box_margin,
        box_margin
    );
    
    vec2 local_capsule_center = (in_center0 + in_center1) * 0.5;
    rr = mat4(1);
    rr = rotate(rr, -vec3(in_angle.x, in_angle.y, angle));
    vec2 vertex_pos = (rr * vec4((rp * (box_size * 0.5)), 0, 1)).xy;
    vec2 world_pos = vertex_pos + local_capsule_center + in_position.xy;
    frag_position = vec3(world_pos, in_position.z);
    
    instance_color = in_color;
    instance_radius = in_radius;
    flags = in_flags;
    
    gl_Position = projection * view * vec4(frag_position, 1.0);
}
