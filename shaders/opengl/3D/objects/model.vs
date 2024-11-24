#version 430 core
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 uv;
layout (location = 3) in ivec4 bone_ids;
layout (location = 4) in vec4 bone_weights;
layout (location = 5) in vec3 tangent;  
layout (location = 6) in vec3 bitangent;

layout(std430, binding = 0) buffer BoneTransforms {
    mat4 bone_transforms[];
};

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;
uniform int use_cpu;

out vec2 tex_coord;
out vec3 v_normal;
out vec3 v_pos;
out vec4 bw;

out vec3 c_tangent;
out vec3 c_bitangent;

void main() {
  if (use_cpu == 0) {
    vec4 totalPosition = vec4(0.0);
    vec3 totalNormal = vec3(0.0);

    for(int i = 0; i < 4; i++) {
      float weight = bone_weights[i];
      int bone_id = bone_ids[i];
      if (bone_id == -1) {
        continue;
      }
        
      if (bone_id >= bone_transforms.length()) {
        totalPosition = vec4(position, 1.0);
        break;
      }
      vec4 local_position = bone_transforms[bone_id] * vec4(position, 1.0);
      totalPosition += local_position * weight;
    }

    vec4 worldPos = model * totalPosition;
    gl_Position = projection * view * worldPos;
    
    //FragPos = vec3(worldPos);
    v_normal = mat3(transpose(inverse(model))) * totalNormal;
    tex_coord = uv;
  }
  else {
	  gl_Position = projection * view * model * vec4(position, 1.0);
    tex_coord = uv;
    v_pos = vec3(model * vec4(position, 1.0));
	  v_normal = mat3(transpose(inverse(model))) * normal;
	  v_normal = normalize(v_normal);
    c_tangent = tangent;
    c_bitangent = bitangent;
  }
}