#version 120
#extension GL_EXT_gpu_shader4 : require

attribute vec3  in_position;
attribute vec3  in_normal;
attribute vec2  in_uv;
attribute ivec4 in_bone_ids;
attribute vec4 in_bone_weights;
attribute vec3 in_tangent;
attribute vec3 in_bitangent;
attribute vec4 in_color;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;
uniform int use_cpu;
uniform int bone_count;

varying vec2 tex_coord;
varying vec3 v_normal;
varying vec3 v_pos;
varying vec4 bw;

varying vec4 vcolor;

varying vec3 c_tangent;
varying vec3 c_bitangent;

uniform mat4 bone_transforms[200];

float det(mat2 matrix) {
    return matrix[0].x * matrix[1].y - matrix[0].y * matrix[1].x;
}

mat3 lowopengl_inverse(mat3 matrix) {
  vec3 row0 = matrix[0];
  vec3 row1 = matrix[1];
  vec3 row2 = matrix[2];

  vec3 minors0 = vec3(
      det(mat2(row1.y, row1.z, row2.y, row2.z)),
      det(mat2(row1.z, row1.x, row2.z, row2.x)),
      det(mat2(row1.x, row1.y, row2.x, row2.y))
  );
  vec3 minors1 = vec3(
      det(mat2(row2.y, row2.z, row0.y, row0.z)),
      det(mat2(row2.z, row2.x, row0.z, row0.x)),
      det(mat2(row2.x, row2.y, row0.x, row0.y))
  );
  vec3 minors2 = vec3(
      det(mat2(row0.y, row0.z, row1.y, row1.z)),
      det(mat2(row0.z, row0.x, row1.z, row1.x)),
      det(mat2(row0.x, row0.y, row1.x, row1.y))
  );

  mat3 adj = transpose(mat3(minors0, minors1, minors2));

  return (1.0 / dot(row0, minors0)) * adj;
}

void main() {
  if (use_cpu == 0) {
    vec4 totalPosition = vec4(0.0);
    vec3 totalNormal = vec3(0.0);
    for(int i = 0; i < 4; i++) {
      float weight = in_bone_weights[i];
      int bone_id = in_bone_ids[i];
      if (weight == 0.0 || bone_id == -1) {
        continue;
      }
      if (bone_id >= bone_count || bone_id >= 200) {
        totalPosition = vec4(in_position, 1.0);
        totalNormal = in_normal;
        break;
      }
            
      mat4 boneTransform = bone_transforms[bone_id];
      vec4 local_position = boneTransform * vec4(in_position, 1.0);
      totalPosition += local_position * weight;
            
      vec3 local_normal = mat3(boneTransform) * in_normal;
      totalNormal += local_normal * weight;
    }

    vec4 worldPos = model * totalPosition;
    gl_Position = projection * view * worldPos;
        
    v_pos = vec3(worldPos);
    v_normal = normalize(mat3(transpose(lowopengl_inverse(mat3(model)))) * totalNormal);
    tex_coord = in_uv;
  } 
  else {
    gl_Position = projection * view * model * vec4(in_position, 1.0);
    tex_coord = in_uv;
    v_pos = vec3(model * vec4(in_position, 1.0));
    v_normal = normalize(mat3(transpose(lowopengl_inverse(mat3(model)))) * in_normal);
  }
  c_tangent = in_tangent;
  c_bitangent = in_bitangent;
  vcolor = in_color;
}
