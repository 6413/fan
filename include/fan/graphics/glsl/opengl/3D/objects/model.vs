R"(

#version 140

in vec4 input0;
in float input1;

uniform mat4 projection;
uniform mat4 view;

mat4 translate(mat4 m, vec3 v) {
  mat4 matrix = m;

  matrix[3][0] = m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2] + m[3][0];
  matrix[3][1] = m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2] + m[3][1];
  matrix[3][2] = m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2] + m[3][2];
  matrix[3][3] = m[0][3] * v[0] + m[1][3] * v[1] + m[2][3] * v[2] + m[3][3];

  return matrix;
}

mat4 scale(mat4 m, vec3 v) {
  mat4 matrix = mat4(1);

  matrix[0][0] = m[0][0] * v[0];
  matrix[0][1] = m[0][1] * v[0];
  matrix[0][2] = m[0][2] * v[0];

  matrix[1][0] = m[1][0] * v[1];
  matrix[1][1] = m[1][1] * v[1];
  matrix[1][2] = m[1][2] * v[1];

  matrix[2][0] = m[2][0] * v[2];
  matrix[2][1] = m[2][1] * v[2];
  matrix[2][2] = m[2][2] * v[2];

  matrix[3][0] = m[3][0];
  matrix[3][1] = m[3][1];
  matrix[3][2] = m[3][2];

  matrix[3] = m[3];

  return matrix;
}

mat4 rotate(mat4 m, float angle, vec3 v) {
  float a = angle;
  float c = cos(a);
  float s = sin(a);
  vec3 axis = vec3(normalize(v));
  vec3 temp = vec3(axis * (1.0f - c));

  mat4 rotation = mat4(1);
  rotation[0][0] = c + temp[0] * axis[0];
  rotation[0][1] = temp[0] * axis[1] + s * axis[2];
  rotation[0][2] = temp[0] * axis[2] - s * axis[1];

  rotation[1][0] = temp[1] * axis[0] - s * axis[2];
  rotation[1][1] = c + temp[1] * axis[1];
  rotation[1][2] = temp[1] * axis[2] + s * axis[0];

  rotation[2][0] = temp[2] * axis[0] + s * axis[1];
  rotation[2][1] = temp[2] * axis[1] - s * axis[0];
  rotation[2][2] = c + temp[2] * axis[2];

  mat4 matrix = mat4(1);
  matrix[0][0] = (m[0][0] * rotation[0][0]) + (m[1][0] * rotation[0][1]) + (m[2][0] * rotation[0][2]);
  matrix[1][0] = (m[0][1] * rotation[0][0]) + (m[1][1] * rotation[0][1]) + (m[2][1] * rotation[0][2]);
  matrix[2][0] = (m[0][2] * rotation[0][0]) + (m[1][2] * rotation[0][1]) + (m[2][2] * rotation[0][2]);

  matrix[0][1] = (m[0][0] * rotation[1][0]) + (m[1][0] * rotation[1][1]) + (m[2][0] * rotation[1][2]);
  matrix[1][1] = (m[0][1] * rotation[1][0]) + (m[1][1] * rotation[1][1]) + (m[2][1] * rotation[1][2]);
  matrix[2][1] = (m[0][2] * rotation[1][0]) + (m[1][2] * rotation[1][1]) + (m[2][2] * rotation[1][2]);

  matrix[0][2] = (m[0][0] * rotation[2][0]) + (m[1][0] * rotation[2][1]) + (m[2][0] * rotation[2][2]);
  matrix[1][2] = (m[0][1] * rotation[2][0]) + (m[1][1] * rotation[2][1]) + (m[2][1] * rotation[2][2]);
  matrix[2][2] = (m[0][2] * rotation[2][0]) + (m[1][2] * rotation[2][1]) + (m[2][2] * rotation[2][2]);

  matrix[3] = m[3];

  return matrix;
}

out vec3 fragment_position;
out vec2 texture_coordinate;
out vec3 normal;
out float is_light;
out vec3 view_position;
out vec3 tanget_fragment_position;
out vec3 tanget_view_position;
out vec3 tanget_light_position;

uniform vec3 p;

uniform mat4 model_input[2];

void main() {

  vec3 layout_vertex = vec3(input0[0], input0[1], input0[2]);
  vec2 layout_texture_coordinate = vec2(input0[3], input1);
  //vec3 layout_normal = vec3(input1[1], input1[2], input1[3]);
  //vec3 layout_tanget = vec3(input2[0], input2[1], input2[2]);
  vec3 layout_position = model_input[0][0].xyz;
  vec3 layout_size = vec3(model_input[0][0].a, model_input[0][1].xy);
  float layout_angle = model_input[0][1].z;
  vec3 layout_rotation_point = vec3(model_input[0][1].a, model_input[0][2].xy);
  vec3 layout_rotation_vector = vec3(model_input[0][2].zw, model_input[0][3].x);
  float layout_is_light = model_input[0][3].y;
  vec3 layout_view_position = vec3(model_input[0][3].z, model_input[0][3].a, model_input[1][0].x);

  mat4 m = mat4(1);
  m = translate(m, layout_position + layout_rotation_point);

  if (!isnan(layout_angle) && !isinf(layout_angle)) {
    vec3 rotation_vector;

    if (layout_rotation_vector.x == 0 && layout_rotation_vector.y == 0 && layout_rotation_vector.z == 0) {
      rotation_vector = vec3(0, 0, 1);
    }
    else {
      rotation_vector = layout_rotation_vector;
    }

    m = rotate(m, layout_angle, rotation_vector);
  }

  m = translate(m, -layout_rotation_point);

  m = scale(m, layout_size);

 // mat3 normal_matrix = transpose(inverse(mat3(m)));
 // vec3 T = normalize(normal_matrix * layout_tanget);
 // vec3 N = normalize(normal_matrix * layout_normal);
 // T = normalize(T - dot(T, N) * N);
 // vec3 B = cross(N, T);
 //   
 // mat3 tbn = transpose(mat3(T, B, N));
 //
  fragment_position = vec3(m * vec4(layout_vertex, 1));
  texture_coordinate = layout_texture_coordinate;
 // normal = mat3(transpose(inverse(m))) * layout_normal;
 // is_light = layout_is_light;
 // view_position = layout_view_position;
 // vec3 light_position = p;
 // tanget_fragment_position = tbn * fragment_position;
 // tanget_view_position = tbn * view_position;
 // tanget_light_position = tbn * light_position;

  gl_Position = projection * view * vec4(fragment_position, 1);
}

)"