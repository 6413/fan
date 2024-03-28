#version 330

in vec2 texture_coordinate;

layout (location = 0) out vec4 o_attachment0;

uniform sampler2D _t00;
uniform sampler2D _t01;
uniform vec3 view_angles;
uniform float tilt;
uniform float zoom;
uniform float depth;
uniform float m_time;
uniform float bloom_strength = 0.004;
uniform float gamma = 2.2f;
uniform float edge0 = 0.3;
uniform float edge1 = 1.0;
uniform float exposure = 1.0f;

#define PI 3.14159265359

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

vec3 quantizeToPalette(vec3 color, vec3[27] palette) {
    // Initialize with the first color in the palette
    vec3 closestColor = palette[0];
    float minDistance = distance(color, closestColor);

    // Iterate over the remaining colors in the palette
    for (int i = 1; i < 27; i++) {
        float dist = distance(color, palette[i]);
        if (dist < minDistance) {
            minDistance = dist;
            closestColor = palette[i];
        }
    }

    return closestColor;
}

vec3 normalizeColor(vec3 color) {
    return color / 255.0;
}


vec3 apply_bloom(vec3 hdrColor, vec3 bloomColor)
{
    // Calculate brightness
    float brightness = dot(bloomColor, vec3(0.2126, 0.7152, 0.0722));

    // Calculate the smoothstep value
    float t = smoothstep(edge0, edge1, bloom_strength * brightness);

    // Mix the hdrColor and bloomColor based on the smoothstep value
    vec3 result = mix(hdrColor, bloomColor, t);

    // Apply exposure and gamma correction
    result = vec3(1.0) - exp(-result * exposure);
    result = pow(result, vec3(1.0 / gamma));

    return result;
}

void main() {
    //vec3 actual = texture(_t00, texture_coordinate).rgb;




        vec2 position = -1.0 + 2.0 * texture_coordinate;
    position *= zoom;

    // Apply tilt (rotate around the x-axis)
    float tilt_angle = tilt;
    float tilt_s = sin(tilt_angle);
    float tilt_c = cos(tilt_angle);
    position = vec2(
        position.x,
        position.y * tilt_c + position.y * tilt_s
    );

    // Apply pitch (rotate around the y-axis)
    float pitch_angle = 0;
    float pitch_s = sin(pitch_angle);
    float pitch_c = cos(pitch_angle);
    position = vec2(
        position.x * pitch_c - position.y * pitch_s,
        position.x * pitch_s + position.y * pitch_c
    );

    // Apply pan
    position += vec2(0, 0.0);

    // Apply spin
    float spin_angle = 0;
    float spin_s = sin(spin_angle);
    float spin_c = cos(spin_angle);
    position = vec2(
        position.x * spin_c - position.y * spin_s,
        position.x * spin_s + position.y * spin_c
    );

    // Normalize position to range [-1, 1]

    // Convert normalized position to spherical coordinates
    float x2y2 = position.x * position.x + position.y * position.y;
    vec3 sphere_pnt = vec3(2.0 * position, x2y2 - 1.0) / (x2y2 + 1.0);

    mat4 m = mat4(1);
    m = rotate(m, view_angles);

    sphere_pnt = vec3(m * vec4(sphere_pnt, 1.0));

    vec2 sampleUV = vec2(
        (atan(sphere_pnt.y, sphere_pnt.x) / PI + 1.0) * 0.5,
        (asin(sphere_pnt.z) / PI + 0.5)
    );

    vec2 b = vec2(1.0 - sampleUV.x, sampleUV.y);
    vec2 a = texture_coordinate;

    // Sample the texture
    vec2 final = vec2(0, 0);
    if (depth > 100) {
      final = b;
    }
    else {
      final = a;
    }
     //vec3 horizontalBlur = blurFunction(_t00, a, vec2(1.0, 0.0), blur_radius);
    
    // Apply vertical blur
   // vec3 verticalBlur = blurFunction(_t00, a, vec2(0.0, 1.0), blur_radius);
   
    // Combine horizontal and vertical blurs
   // vec3 finalColor = (horizontalBlur + verticalBlur) / 2.0;
   // finalColor = blurFunction(_t00, a, vec2(0.0, 1.0), blur_radius);
   //const float gamma = 1;
   // vec3 finalColor = applyBloom(_t00, a, actual);
    //finalColor = vec3(1.0) - exp(-finalColor * exposure);
   // finalColor = pow(finalColor, vec3(1.0 / gamma));

  vec3 hdrColor = texture(_t00, texture_coordinate).rgb;
  vec3 bloomColor = texture(_t01, texture_coordinate).rgb;
  o_attachment0 = vec4(apply_bloom(hdrColor, bloomColor), 1.0);
  //o_attachment0 = vec4(hdrColor, 1.0);
}