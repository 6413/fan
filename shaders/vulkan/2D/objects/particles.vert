#version 450

struct particle_data_t {
  vec4 position_shape;
  vec4 loop_times;
  vec4 count_life;
  vec4 size;
  vec4 color0;
  vec4 color1;
  vec4 velocity;
  vec4 angle_velocity0;
  vec4 angle_velocity1;
  vec4 angle;
  vec4 spawn_spread0;
  vec4 spread1_jitter;
  vec4 jitter_random_size;
  vec4 color_random;
  vec4 angle_random;
};

layout(std430, set = 0, binding = 0) readonly buffer particle_t {
  particle_data_t particles[];
};

layout(push_constant) uniform constants_t {
  uint texture_id;
  uint camera_id;
  uint texture_id1;
  uint texture_id2;
  uint texture_id3;
} constants;

struct pv_t {
  mat4 projection;
  mat4 view;
};

layout(set = 0, binding = 1) uniform upv_t {
  pv_t pv[16];
};

vec2 tc[6] = vec2[](
  vec2(0.0, 0.0),
  vec2(1.0, 0.0),
  vec2(1.0, 1.0),
  vec2(1.0, 1.0),
  vec2(0.0, 1.0),
  vec2(0.0, 0.0)
);

vec2 rectangle_vertices[6] = vec2[](
  vec2(-1.0, -1.0),
  vec2(1.0, -1.0),
  vec2(1.0, 1.0),
  vec2(1.0, 1.0),
  vec2(-1.0, 1.0),
  vec2(-1.0, -1.0)
);

uint rand_u(uint seed) {
  for (uint i = 0u; i < 2u; ++i) {
    seed += seed << 10u;
    seed ^= seed >> 6u;
    seed += seed << 3u;
    seed ^= seed >> 11u;
    seed += seed << 15u;
  }
  return seed;
}

float rand_f(uint seed) {
  uint m = rand_u(seed);
  m &= 0x007fffffu;
  m |= 0x3f800000u;
  return uintBitsToFloat(m) - 1.0f;
}

layout(location = 0) out vec4 instance_color;
layout(location = 1) out vec2 texture_coordinate;
layout(location = 2) out float affected_by_lighting;

void main() {
  particle_data_t p = particles[gl_InstanceIndex];
  uint count = max(uint(p.count_life.x), 1u);
  uint particle_id = uint(gl_VertexIndex) / 6u + 1u;
  if (particle_id > count) { gl_Position = vec4(0.0); return; }
  uint vertex_id = uint(gl_VertexIndex) % 6u;
  uint seed = particle_id * count;
  float alive_time = max(p.count_life.y, 0.0001);
  float respawn_time = p.count_life.z;
  float cycle = alive_time + respawn_time;
  float rand_val = rand_f(seed);
  bool loop = p.loop_times.x > 0.5;
  float loop_enabled_time = p.loop_times.y;
  float loop_disabled_time = p.loop_times.z;
  float time = p.loop_times.w;
  float new_time = 0.0;
  float time_mod = 0.0;
  float spawn_delay = rand_val * cycle;

  if (loop) {
    float local_time = time - loop_enabled_time;
    if (loop_disabled_time > 0.0) {
      float time_at_disable = loop_disabled_time - loop_enabled_time;
      if (time_at_disable < spawn_delay) { instance_color = vec4(0.0); gl_Position = vec4(0.0); return; }
      float time_since_stop = time - loop_disabled_time;
      float age_at_stop = mod(time_at_disable - spawn_delay, cycle) - respawn_time;
      if (age_at_stop < 0.0) { instance_color = vec4(0.0); gl_Position = vec4(0.0); return; }
      new_time = age_at_stop + time_since_stop;
      time_mod = new_time;
      if (new_time > alive_time) { instance_color = vec4(0.0); gl_Position = vec4(0.0); return; }
    }
    else {
      if (local_time <= 0.0001 || local_time < spawn_delay) { instance_color = vec4(0.0); gl_Position = vec4(0.0); return; }
      new_time = local_time - spawn_delay;
      time_mod = mod(new_time, cycle) - respawn_time;
    }
  }
  else {
    float local_time = time - loop_enabled_time;
    if (local_time < 0.0 || local_time < spawn_delay) { instance_color = vec4(0.0); gl_Position = vec4(0.0); return; }
    new_time = local_time - spawn_delay;
    if (new_time > alive_time) { instance_color = vec4(0.0); gl_Position = vec4(0.0); return; }
    time_mod = new_time;
  }

  if (time_mod < 0.0) { instance_color = vec4(0.0); gl_Position = vec4(0.0); return; }

  float t = clamp(time_mod / alive_time, 0.0, 1.0);
  uint pseed = seed + particle_id * 7919u;
  vec2 origin = p.position_shape.xy;
  vec2 base_pos = origin;
  vec2 start_spread = p.spawn_spread0.zw;
  vec2 end_spread = p.spread1_jitter.xy;
  vec2 spread_max = mix(start_spread, end_spread, t);

  if (int(p.position_shape.w) == 1) {
    base_pos.x += (rand_f(particle_id * 7919u) - 0.5) * spread_max.x;
    base_pos.y += (rand_f(particle_id * 7919u + 1u) - 0.5) * spread_max.y;
  }
  else {
    float ang = rand_f(particle_id * 7919u) * 6.28318530718;
    float r = rand_f(particle_id * 7919u + 1u);
    base_pos += vec2(cos(ang) * spread_max.x * r, sin(ang) * spread_max.y * r);
  }

  vec2 avg_vel_mag = p.velocity.xy + (p.velocity.zw - p.velocity.xy) * 0.5 * t;
  float spread = mix(p.angle_velocity0.w, p.angle_velocity1.w, rand_f(seed + 2u));
  float ca = cos(spread);
  float sa = sin(spread);
  vec2 avg_velocity = vec2(avg_vel_mag.x * ca - avg_vel_mag.y * sa, avg_vel_mag.x * sa + avg_vel_mag.y * ca);
  base_pos += avg_velocity * pow(time_mod, p.count_life.w);

  vec2 jitter_amount = mix(p.jitter_random_size.xy, p.jitter_random_size.zw, t);
  float jitter_speed = p.spread1_jitter.z;
  base_pos += vec2(sin(time_mod * jitter_speed + float(seed)) * jitter_amount.x, cos(time_mod * jitter_speed + float(seed)) * jitter_amount.y);

  vec2 size = mix(p.size.xy, p.size.zw, t);
  if (p.angle_random.w > 0.0) {
    float size_rand = rand_f(pseed * 2u);
    size *= 1.0 + (size_rand - 0.5) * p.angle_random.w * 2.0;
  }

  vec3 angle_vel = mix(p.angle_velocity0.xyz, p.angle_velocity1.xyz, t);
  vec3 total_angle = p.angle.xyz + time_mod * angle_vel;
  if (length(p.angle_random.xyz) > 0.0) {
    vec3 rand_angle = vec3(rand_f(pseed * 13u), rand_f(pseed * 17u), rand_f(pseed * 19u));
    total_angle += (rand_angle - 0.5) * p.angle_random.xyz * 2.0;
  }

  float sx = sin(total_angle.x);
  float cx = cos(total_angle.x);
  float sy = sin(total_angle.y);
  float cy = cos(total_angle.y);
  float sz = sin(total_angle.z);
  float cz = cos(total_angle.z);

  vec2 offset = base_pos - origin;
  offset = vec2(offset.x * cz - offset.y * sz, offset.x * sz + offset.y * cz);

  vec3 v = vec3(rectangle_vertices[vertex_id] * size, 0.0);
  v = vec3(
    cy * cz * v.x + (sx * sy * cz - cx * sz) * v.y,
    cy * sz * v.x + (sx * sy * sz + cx * cz) * v.y,
    -sy * v.x + sx * cy * v.y
  );

  vec2 world_pos = origin + offset + v.xy;
  gl_Position = pv[constants.camera_id].projection * pv[constants.camera_id].view * vec4(world_pos, p.position_shape.z + v.z, 1.0);
  gl_Position.z += (float(particle_id) / float(count)) * 0.0001;

  vec4 color = mix(p.color0, p.color1, t);
  float fade_in = smoothstep(0.0, 0.08, t);
  color.a *= fade_in;
  if (length(p.color_random) > 0.0) {
    vec4 rand_color = vec4(rand_f(pseed * 3u), rand_f(pseed * 5u), rand_f(pseed * 7u), rand_f(pseed * 11u));
    color = clamp(color + (rand_color - 0.5) * p.color_random * 2.0, 0.0, 1.0);
  }

  instance_color = color;
  texture_coordinate = tc[vertex_id];
  affected_by_lighting = p.spread1_jitter.w;
}
