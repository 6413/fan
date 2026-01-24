#version 330

out vec4 i_color;

uniform bool loop;
uniform float loop_enabled_time;
uniform float loop_disabled_time;
uniform uint vertex_count;
uniform uint count;

uniform vec2 position;

uniform vec2 start_size;
uniform vec2 end_size;

uniform vec2 start_velocity;
uniform vec2 end_velocity;

uniform vec3 start_angle_velocity;
uniform vec3 end_angle_velocity;

uniform float alive_time;
uniform float respawn_time;

uniform float begin_angle;
uniform float end_angle;

uniform vec3 angle;

uniform vec4 begin_color;
uniform vec4 end_color;

uniform vec2 spawn_spacing;
uniform vec2 start_spread;
uniform vec2 end_spread;
uniform float expansion_power;

uniform vec2 jitter_start;
uniform vec2 jitter_end;
uniform float jitter_speed;

uniform vec2 size_random_range;
uniform vec4 color_random_range;
uniform vec3 angle_random_range;

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

vec2 vec2_direction(uint r, uint r2, float min, float max){
  float rr = mod(2.0 * 3.141 * floatConstruct(r), -max) - min;
  float rr2 = mod(2.0 * 3.141 * floatConstruct(r2), -max) - min;
  return vec2(cos(rr), sin(rr2));
}

void main() {
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
        i_color = vec4(0);
        return;
      }

      float time_since_stop = time - loop_disabled_time;
      float age_at_stop = mod(time_at_disable - spawn_delay, cycle) - respawn_time;

      if (age_at_stop < 0.0) {
        i_color = vec4(0);
        return;
      }

      new_time = age_at_stop + time_since_stop;
      time_mod = new_time;

      if (new_time > alive_time) {
        i_color = vec4(0);
        return;
      }
    } else {
      if (local_time <= 0.0001 || local_time < spawn_delay) {
        i_color = vec4(0);
        return;
      }

      new_time = local_time - spawn_delay;
      time_mod = mod(new_time, cycle) - respawn_time;
    }
  } else {
    float spawn_delay = rand_val * cycle;

    if (loop_disabled_time > 0.0) {
      float local_time_at_disable = loop_disabled_time - loop_enabled_time;
      float time_since_disable = time - loop_disabled_time;

      if (local_time_at_disable < spawn_delay) {
        i_color = vec4(0);
        return;
      }

      new_time = (local_time_at_disable - spawn_delay) + time_since_disable;

      if (new_time > alive_time) {
        i_color = vec4(0);
        return;
      }
    } else {
      float local_time = time - loop_enabled_time;

      if (local_time <= 0.0001 || local_time < spawn_delay) {
        i_color = vec4(0);
        return;
      }

      new_time = local_time - spawn_delay;

      if (new_time > alive_time) {
        i_color = vec4(0);
        return;
      }
    }

    time_mod = new_time;
  }

  if (time_mod < 0.0) {
    i_color = vec4(0);
    return;
  }

  float t = clamp(time_mod / alive_time, 0.0, 1.0);

  uint pseed = seed + id * 7919u;

  vec2 base_pos = position;
  vec2 spread_max = mix(start_spread, end_spread, t);

  if (shape == 1) {
    base_pos.x += (floatConstruct(RAND(id * 7919u)) - 0.5) * spread_max.x;
    base_pos.y += (floatConstruct(RAND(id * 7919u + 1u)) - 0.5) * spread_max.y;
  } 
  else {
    float ang = floatConstruct(RAND(id * 7919u)) * 6.28318530718;
    float r = floatConstruct(RAND(id * 7919u + 1u));
    base_pos += vec2(cos(ang) * spread_max.x * r,
                     sin(ang) * spread_max.y * r);
  }

  vec2 dir = vec2_direction(RAND(seed + 2u), RAND(seed + 3u), begin_angle, end_angle);
  dir = normalize(dir);

  vec2 vel_mag = mix(start_velocity, end_velocity, t);
  vec2 velocity = dir * vel_mag;

  float expansion = pow(time_mod, expansion_power);
  base_pos += velocity * expansion;

  vec2 jitter_amount = mix(jitter_start, jitter_end, t);
  base_pos += vec2(
    sin(time_mod * jitter_speed + float(seed)) * jitter_amount.x,
    cos(time_mod * jitter_speed + float(seed)) * jitter_amount.y
  );

  vec2 size;
  if (size_random_range.x > 0.0) {
    float size_rand = floatConstruct(RAND(pseed * 2u));
    size = mix(start_size, end_size, t) * (1.0 + (size_rand - 0.5) * size_random_range.x * 2.0);
  }
  else {
    size = mix(start_size, end_size, t);
  }

  vec3 angle_vel = mix(start_angle_velocity, end_angle_velocity, t);
  vec3 total_angle;
  if (angle_random_range.x > 0.0) {
    vec3 rand_angle = vec3(
      floatConstruct(RAND(pseed * 13u)),
      floatConstruct(RAND(pseed * 17u)),
      floatConstruct(RAND(pseed * 19u))
    );
    total_angle = angle + time_mod * angle_vel + (rand_angle - 0.5) * angle_random_range * 2.0;
  }
  else {
    total_angle = angle + time_mod * angle_vel;
  }

  vec2 v = rectangle_vertices[gl_VertexID % 6] * size;

  float c = cos(total_angle.z);
  float s = sin(total_angle.z);

  vec2 offset = base_pos - position;
  offset = vec2(
    offset.x * c - offset.y * s,
    offset.x * s + offset.y * c
  );

  vec2 world_pos = position + offset + v;

  gl_Position = projection * view * vec4(world_pos, 0.0, 1.0);
  gl_Position.z = 1.0 - (float(modded_index) / 6.0) / float(count);

  vec4 particle_color;
  if (color_random_range.x > 0.0) {
    vec4 rand_color = vec4(
      floatConstruct(RAND(pseed * 3u)),
      floatConstruct(RAND(pseed * 5u)),
      floatConstruct(RAND(pseed * 7u)),
      floatConstruct(RAND(pseed * 11u))
    );
    particle_color = mix(begin_color, end_color, t);
    particle_color += (rand_color - 0.5) * color_random_range * 2.0;
    particle_color = clamp(particle_color, 0.0, 1.0);
  }
  else {
    particle_color = mix(begin_color, end_color, t);
  }

  i_color = particle_color;
  texture_coordinate = tc[gl_VertexID % 6];
}