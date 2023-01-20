R"(
#version 130

in vec4 instance_color;
in vec2 instance_position;
in vec2 instance_fragment_position;
in float instance_radius;
in mat4 mvp;

out vec4 color;

void main() {
  vec4 transformedCoord = mvp * vec4(gl_FragCoord.xy, 0, 1.0);
  //transformedCoord.xy += vec2(100);
  //coord += vec2(100);
  float dist = length(transformedCoord.xy - instance_position);
//color = instance_color;
  if (dist < instance_radius) {
    // fragment is inside the circle
    // set the color of your choice
    color = instance_color;
  } else {
    // fragment is outside the circle
    // set the color of your choice
    color = vec4(instance_color.rgb, 0);
  }
  //float distance = distance(instance_fragment_position, gl_FragCoord.xy);
  //if (distance < instance_radius) {
  //  const float smoothness = 2;
  //  //float a = smoothstep(0, 1, (instance_radius - distance) / smoothness);
  //  color = instance_color;
  //}
  //else {
  //  color = vec4(instance_color.rgb, 0);
  //} 
}
)"