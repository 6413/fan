#include <fan/pch.h>

// interpolate between two colors
fan::color interpolate(const fan::color& c1, const fan::color& c2, f32_t factor) {
  fan::color r;
  r.r = c1.r + (c2.r - c1.r) * factor;
  r.g = c1.g + (c2.g - c1.g) * factor;
  r.b = c1.b + (c2.b - c1.b) * factor;
  return r;
}


constexpr fan::color get_height_color(int value) {
  constexpr std::pair<int, fan::color> color_map[] = {
    { 50,   fan::color::from_rgba(0x003eb2ff)},     // Dark blue for deep water
    { 80,  fan::color::from_rgba(0x0952c6ff)}, // Light blue for shallow water
    { 100,  fan::color::from_rgba(0x726231ff)},   // Brown for coastal areas
    { 150, fan::color::from_rgba(0xa49463ff)}, // Tan for lowlands
    { 200, fan::color::from_rgba(0x3c6114ff)},     // Dark green for midlands
    { 250, fan::color::from_rgba(0x4f6b31ff)}, // Light green for lower highlands
    { 300, fan::color::from_rgba(0xffffffff)}, // Snow at the highest elevations
  };

  int n = sizeof(color_map) / sizeof(color_map[0]);

  // The smoothingFactor and value adjustment remain the same
  const float smoothingFactor = 1;
  value = static_cast<int>(value * smoothingFactor + (1 - smoothingFactor) * value);

  // Handle values outside the expected range
  if (value < color_map[0].first) return color_map[0].second;
  if (value > color_map[n - 1].first) return color_map[n - 1].second;

  for (int i = 0; i < n - 1; ++i) {
    if (value >= color_map[i].first && value <= color_map[i + 1].first) {
      int v1 = color_map[i].first;
      int v2 = color_map[i + 1].first;
      fan::color c1 = color_map[i].second;
      fan::color c2 = color_map[i + 1].second;

      float factor = (value - v1) / float(v2 - v1);
      return c1;
    }
  }

  // Fallback for any value not handled by the loop
  return fan::color::rgb(0, 0, 0);
}

void generate_mesh(loco_t& loco, fan::vec2& noise_size, std::vector<uint8_t>& noise_image_data, const fan::opengl::context_t::image_nr_t& dirt, std::vector<loco_t::shape_t>& built_mesh)
{
  loco_t::sprite_t::properties_t sp;
  sp.size = loco.window.get_size() / noise_size / 2;
  int idx = 0;
  for (int i = 0; i < noise_size.y; ++i) {
    for (int j = 0; j < noise_size.x; ++j) {
      int index = (i * noise_size.x + j) * 4;
      int r = noise_image_data[index];

      int grayscale = r;
      sp.position = fan::vec2(i, j) * sp.size * 2;
      sp.image = dirt;
      sp.color = get_height_color(grayscale);
      sp.color.a = 1;
      built_mesh.push_back(sp);
      idx++;
    }
  }
}

std::vector<uint8_t> generate_noise(FastNoiseLite& fn, const fan::vec2& noise_size, f32_t noise_tex_min, f32_t noise_tex_max) {
  std::vector<uint8_t> noise_data(noise_size.multiply() * 4);

  int index = 0;

  float scale = 255 / (noise_tex_max - noise_tex_min);

  for (int y = 0; y < noise_size.y; y++)
  {
    for (int x = 0; x < noise_size.x; x++)
    {
      float noiseValue = fn.GetNoise((float)x, (float)y);
      unsigned char cNoise = (unsigned char)std::max(0.0f, std::min(255.0f, (noiseValue - noise_tex_min) * scale));
      noise_data[index * 4 + 0] = cNoise;
      noise_data[index * 4 + 1] = cNoise;
      noise_data[index * 4 + 2] = cNoise;
      noise_data[index * 4 + 3] = 255;
      index++;
    }
  }
  return noise_data;
}

int main() {
  loco_t loco{ {.window_size = 1024} };

  auto dirt = loco.create_image(fan::colors::white);
  auto water = loco.image_load("images/water.webp");

  fan::vec2 noise_size = 512;
  auto noise_image_data = loco.create_noise_image_data(noise_size);

  std::vector<loco_t::shape_t> built_mesh;

  generate_mesh(loco, noise_size, noise_image_data, dirt, built_mesh);


  int seed = fan::random::value_i64(0, ((uint32_t)-1) / 2);


  FastNoiseLite noise;
  noise.SetFractalType(FastNoiseLite::FractalType_FBm);
  noise.SetNoiseType(FastNoiseLite::NoiseType_OpenSimplex2);
  float noise_frequency = 0.010;
  float noise_fractal_gain = 0.5;
  float noise_fractal_lacunarity = 2.0;
  int noise_fractal_octaves = 5;

  noise.SetFractalPingPongStrength(2.0);
  f32_t noise_tex_min = -1;
  f32_t noise_tex_max = 0.1;

  bool reload_mesh = true;

  std::vector<uint8_t> noise_data;

  loco.loop([&] {
    if (reload_mesh) {

      noise.SetFrequency(noise_frequency);
      noise.SetFractalGain(noise_fractal_gain);
      noise.SetFractalLacunarity(noise_fractal_lacunarity);
      noise.SetFractalOctaves(noise_fractal_octaves);
      noise.SetSeed(seed);

      built_mesh.clear();
      noise_data.clear();
      noise_data = generate_noise(noise, noise_size, noise_tex_min, noise_tex_max);
      generate_mesh(loco, noise_size, noise_data, dirt, built_mesh);
      reload_mesh = false;
    }

    ImGui::Begin("World settings");

    reload_mesh =
      ImGui::DragInt("seed", &seed, 0.1) |
      ImGui::DragInt("octaves", &noise_fractal_octaves, 1) |
      ImGui::DragFloat("frequency", &noise_frequency, 0.001) |
      ImGui::DragFloat("fractal gain", &noise_fractal_gain, 0.01) |
      ImGui::DragFloat("fractal lacunarity", &noise_fractal_lacunarity, 0.01) |
      ImGui::DragFloat("texture min", &noise_tex_min, 0.01) |
      ImGui::DragFloat("texture max", &noise_tex_max, 0.01);

    ImGui::End();

    });
}