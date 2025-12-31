#version 330

out vec4 i_color;

uniform bool loop;
uniform float loop_enabled_time;
uniform float loop_disabled_time;
uniform uint vertex_count;
uniform uint count;
uniform vec2 position;
uniform vec2 size;
uniform vec2 position_velocity;
uniform vec3 angle_velocity;
uniform float alive_time;
uniform float respawn_time;
uniform float begin_angle;
uniform float end_angle;
uniform vec3 angle;
uniform vec4 begin_color;
uniform vec4 end_color;
uniform vec2 gap_size;
uniform vec2 max_spread_size;
uniform float expansion_power;
uniform vec2 size_velocity;
uniform vec2 turbulence;
uniform float turbulence_speed;
uniform int shape;
uniform float time;
uniform mat4 projection;
uniform mat4 view;

out vec2 texture_coordinate;

vec2 tc[] = vec2[](
  vec2(0, 0),
  vec2(1, 0),
  vec2(1, 1),
  vec2(1, 1),
  vec2(0, 1),
  vec2(0, 0)
);

vec2 triangle_vertices[] = vec2[](
  vec2(0.0, (2.0 * sqrt(3.0)) / 6.0),
  vec2(-1.0 / 2.0, -sqrt(3.0) / 6.0),
  vec2(1.0 / 2.0, -sqrt(3.0) / 6.0)
);

vec2 rectangle_vertices[] = vec2[](
  vec2(-1.0, -1.0),
  vec2(1.0, -1.0),
  vec2(1.0, 1.0),
  vec2(1.0, 1.0),
  vec2(-1.0, 1.0),
  vec2(-1.0, -1.0)
);

uint RAND(uint seed){
  for (uint i = 0u; i < 2u; i++) {
    seed += (seed << 10u);
    seed ^= (seed >> 6u);
    seed += (seed << 3u);
    seed ^= (seed >> 11u);
    seed += (seed << 15u);
  }
  return seed;
}
float floatConstruct(uint m){
  const uint ieeeMantissa = 0x007FFFFFu;
  const uint ieeeOne = 0x3F800000u;
  m &= ieeeMantissa;
  m |= ieeeOne;
  return uintBitsToFloat(m) - 1.0;
}
mat4 translate(mat4 m, vec3 v){
  mat4 matrix = m;
  matrix[3][0] = m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2] + m[3][0];
  matrix[3][1] = m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2] + m[3][1];
  matrix[3][2] = m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2] + m[3][2];
  matrix[3][3] = m[0][3] * v[0] + m[1][3] * v[1] + m[2][3] * v[2] + m[3][3];
  return matrix;
}
mat4 scale(mat4 m, vec3 v){
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
  matrix[3] = m[3];
  return matrix;
}
mat4 rotate(mat4 m, vec3 angles){
  float cx = cos(angles.x);
  float sx = sin(angles.x);
  float cy = cos(angles.y);
  float sy = sin(angles.y);
  float cz = cos(angles.z);
  float sz = sin(angles.z);
  mat4 rotationX = mat4(1.0, 0.0, 0.0, 0.0, 0.0, cx, -sx, 0.0, 0.0, sx, cx, 0.0, 0.0, 0.0, 0.0, 1.0);
  mat4 rotationY = mat4(cy, 0.0, sy, 0.0, 0.0, 1.0, 0.0, 0.0, -sy, 0.0, cy, 0.0, 0.0, 0.0, 0.0, 1.0);
  mat4 rotationZ = mat4(cz, -sz, 0.0, 0.0, sz, cz, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
  return rotationX * rotationY * rotationZ * m;
}
vec2 vec2_direction(uint r, uint r2, float min, float max){
  float rr = mod(2.0 * 3.141 * floatConstruct(r), -max) - min;
  float rr2 = mod(2.0 * 3.141 * floatConstruct(r2), -max) - min;
  return vec2(cos(rr), sin(rr2));
}

// dont look here
void main(){
  int modded_index = gl_VertexID % (int(count) * 6);
  uint id = uint(gl_VertexID) / vertex_count + 1u;
  uint seed = id * count;
  float cycle = alive_time + respawn_time;
  float rand_val = floatConstruct(RAND(seed));
  
  float new_time;
  float time_mod;
  
  if (loop) {
    float spawn_delay = rand_val * cycle;
    float local_time = time - loop_enabled_time;
    
    if (loop_disabled_time > 0.0) {
      float time_at_disable = loop_disabled_time - loop_enabled_time;
      
      if (time_at_disable < spawn_delay) {
        i_color = vec4(0, 0, 0, 0);
        return;
      }
      
      float time_since_stop = time - loop_disabled_time;
      float age_at_stop = time_at_disable - spawn_delay;
      age_at_stop = mod(age_at_stop, cycle) - respawn_time;
      
      if (age_at_stop < 0.0) {
        i_color = vec4(0, 0, 0, 0);
        return;
      }
      
      new_time = age_at_stop + time_since_stop;
      time_mod = new_time;
      
      if (new_time > alive_time) {
        i_color = vec4(0, 0, 0, 0);
        return;
      }
    }
    else {
      if (local_time <= 0.0001 || local_time < spawn_delay) {
        i_color = vec4(0, 0, 0, 0);
        return;
      }
      
      new_time = local_time - spawn_delay;
      time_mod = mod(new_time, cycle) - respawn_time;
    }
  }
  else {
    if (loop_disabled_time > 0.0) {
      float local_time_at_disable = loop_disabled_time - loop_enabled_time;
      float spawn_delay = rand_val * cycle;
      float time_since_disable = time - loop_disabled_time;
      
      if (local_time_at_disable < spawn_delay) {
        i_color = vec4(0, 0, 0, 0);
        return;
      }
      
      new_time = (local_time_at_disable - spawn_delay) + time_since_disable;
      
      if (new_time > alive_time) {
        i_color = vec4(0, 0, 0, 0);
        return;
      }
    }
    else {
      float spawn_delay = rand_val * cycle;
      float local_time = time - loop_enabled_time;
  
      if (local_time <= 0.0001 || local_time < spawn_delay) {
        i_color = vec4(0, 0, 0, 0);
        return;
      }
  
      new_time = local_time - spawn_delay;
  
      if (new_time > alive_time) {
        i_color = vec4(0, 0, 0, 0);
        return;
      }
    }
    
    time_mod = new_time;
  }

  if (time_mod < 0.0) {
    i_color = vec4(0, 0, 0, 0);
    return;
  }

  if (!loop && loop_disabled_time > 0.0 && time - time_mod - respawn_time > loop_disabled_time) {
    i_color = vec4(0, 0, 0, 0);
    return;
  }

  vec2 pos = position;
  if (shape == 1) {
    pos.x += (floatConstruct(RAND(id * 7919u)) - 0.5) * max_spread_size.x;
    pos.y += (floatConstruct(RAND(id * 7919u + 1u)) - 0.5) * max_spread_size.y;
  }
  else {
    float ang = floatConstruct(RAND(id * 7919u)) * 6.28318;
    float radius = floatConstruct(RAND(id * 7919u + 1u)) * max_spread_size.x;
    pos += vec2(cos(ang), sin(ang)) * radius;
  }

  vec2 velocity = vec2_direction(RAND(seed + 2u), RAND(seed + 3u), begin_angle, end_angle);
  float lenv = length(velocity);
  if (lenv != 0.0) {
    velocity /= lenv;
  }
  velocity *= position_velocity;

  float expansion = pow(time_mod, expansion_power);
  pos += velocity * expansion;

  

  pos += vec2(
    sin(time_mod * turbulence_speed + float(seed)) * turbulence.x,
    cos(time_mod * turbulence_speed + float(seed)) * turbulence.y
  );


  mat4 m = translate(mat4(1), vec3(pos, 0.0));
  m = rotate(m, angle + time_mod * angle_velocity);
  m = scale(m, vec3(size * (vec2(1.0) + size_velocity * time_mod), 0.0));

  gl_Position = projection * view * m * vec4(rectangle_vertices[gl_VertexID % 6], 0, 1);
  gl_Position.z = 1.0 - (float(modded_index) / 6.0) / float(count);

  float t = clamp(time_mod / alive_time, 0.0, 1.0);
  vec4 c = mix(begin_color, end_color, t);
  i_color = c;

  texture_coordinate = tc[gl_VertexID % 6];
}