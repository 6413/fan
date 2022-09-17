#pragma once

// compile with MD

#include _FAN_PATH(types/types.h)

#include _FAN_PATH(graphics/graphics.h)

#include <vpx/vpx_decoder.h)
#include <vpx/vp8dx.h)

#pragma comment(lib, "Onecore.lib")
#pragma comment(lib, "lib/libvpx/vpxmd.lib")
#pragma comment(lib, "lib/libvpx/vpxrcmd.lib")
#pragma comment(lib, "lib/libvpx/gtestmd.lib")

vpx_codec_ctx_t* init_decoder(int width, int height, const char* colorspace)
{
  int flags = 0;
  int err = 0;
  vpx_codec_iface_t* codec_iface = nullptr;
  vpx_codec_ctx_t* ctx = (vpx_codec_ctx_t*)malloc(sizeof(vpx_codec_ctx_t));
  if (ctx == nullptr)
    return nullptr;
  memset(ctx, 0, sizeof(vpx_codec_ctx_t));
  err = vpx_codec_dec_init(ctx, codec_iface, nullptr, flags);
  if (err) {
    fan::throw_error("vpx_codec_dec_init(..) failed with error " + std::to_string(err));
    free(ctx);
    return nullptr;
  }
  return ctx;
}

#define IVF_FRAME_HDR_SZ (4 + 8) /* 4 byte size + 8 byte timestamp */

static uint32_t mem_get_le16(const void* vmem) {
  uint32_t val;
  const uint8_t* mem = (const uint8_t*)vmem;

  val = mem[1] << 8;
  val |= mem[0];
  return val;
}

static uint32_t mem_get_le32(const void* vmem) {
  uint32_t val;
  const uint8_t* mem = (const uint8_t*)vmem;

  val = ((uint8_t)mem[3]) << 24;
  val |= mem[2] << 16;
  val |= mem[1] << 8;
  val |= mem[0];
  return val;
}

int ivf_read_frame(FILE* infile, uint8_t** buffer, size_t* bytes_read,
  size_t* buffer_size) {
  char raw_header[IVF_FRAME_HDR_SZ] = { 0 };
  size_t frame_size = 0;

  if (fread(raw_header, IVF_FRAME_HDR_SZ, 1, infile) != 1) {
    if (!feof(infile)) fan::print_warning("failed to read frame size");
  }
  else {
    frame_size = mem_get_le32(raw_header);

    if (frame_size > 256 * 1024 * 1024) {
      fan::print_warning("read invalid frame size " + std::to_string((unsigned int)frame_size));
      frame_size = 0;
    }

    if (frame_size > *buffer_size) {
      uint8_t* new_buffer = (uint8_t*)realloc(*buffer, 2 * frame_size);

      if (new_buffer) {
        *buffer = new_buffer;
        *buffer_size = 2 * frame_size;
      }
      else {
        fan::print_warning("failed to allocate compressed data buffer");
        frame_size = 0;
      }
    }
  }

  if (!feof(infile)) {
    if (fread(*buffer, 1, frame_size, infile) != frame_size) {
      fan::print_warning("failed to read full frame");
      return 1;
    }

    *bytes_read = frame_size;
    return 0;
  }

  return 1;
}

struct video_info_t {
  uint32_t codec_fourcc;
  fan::vec2ui size;
  uint16_t frame_rate;
};

struct video_reader_t {
  video_info_t info;
  FILE* file;
  uint8_t* buffer;
  size_t buffer_size;
  size_t frame_size;
};

static constexpr const char* const kIVFSignature = "DKIF";

video_reader_t* vpx_video_reader_open(const char* filename) {
  char header[32];
  video_reader_t* reader = nullptr;
  FILE* const file = fopen(filename, "rb");
  if (!file) {
    fprintf(stderr, "%s can't be opened.\n", filename);  // Can't open file
    return NULL;
  }

  if (fread(header, 1, 32, file) != 32) {
    fprintf(stderr, "File header on %s can't be read.\n",
      filename);  // Can't read file header
    return NULL;
  }
  if (memcmp(kIVFSignature, header, 4) != 0) {
    fprintf(stderr, "The IVF signature on %s is wrong.\n",
      filename);  // Wrong IVF signature

    return NULL;
  }
  if (mem_get_le16(header + 4) != 0) {
    fprintf(stderr, "%s uses the wrong IVF version.\n",
      filename);  // Wrong IVF version

    return NULL;
  }

  reader = (video_reader_t*)calloc(1, sizeof(*reader));
  if (!reader) {
    fprintf(
      stderr,
      "Can't allocate VpxVideoReader\n");  // Can't allocate VpxVideoReader

    return NULL;
  }

  reader->file = file;
  reader->info.codec_fourcc = mem_get_le32(header + 8);
  reader->info.size.x = mem_get_le16(header + 12);
  reader->info.size.y = mem_get_le16(header + 14);
  reader->info.frame_rate = mem_get_le32(header + 16);

  return reader;
}

int vpx_video_reader_read_frame(video_reader_t* reader) {
  return !ivf_read_frame(reader->file, &reader->buffer, &reader->frame_size,
    &reader->buffer_size);
}

void vpx_video_reader_close(video_reader_t* reader) {
  if (reader) {
    fclose(reader->file);
    free(reader->buffer);
    free(reader);
  }
}

const uint8_t* vpx_video_reader_get_frame(video_reader_t* reader, size_t* size) {
  *size = reader->frame_size;
  return reader->buffer;
}

fan::vec2ui vpx_get_plane_size(const vpx_image_t* img, int plane) {

  fan::vec2ui size;

  if (plane > 0 && img->x_chroma_shift > 0)
    size.x = (img->d_w + 1) >> img->x_chroma_shift;
  else
    size.x = img->d_w;

  if (plane > 0 && img->y_chroma_shift > 0)
    size.y = (img->d_h + 1) >> img->y_chroma_shift;
  else
    size.y = img->d_h;

  return size;
}

#define VP8_FOURCC 0x30385056
#define VP9_FOURCC 0x30395056

namespace fan_2d {
  namespace opengl {

    struct video_t {

      struct properties_t {

      };

      struct out_t {
        out_t() : pixel_data() {}
        ~out_t() {
          for (int i = 0; i < 4; i++) {
            if (pixel_data.pixels[i] != nullptr) {
              delete pixel_data.pixels[i];
            }
          }
        }

        fan_2d::opengl::pixel_data_t pixel_data;
      };

      uint64_t m_current_delta;
      uint64_t m_next_delta;

      vpx_codec_ctx_t* codec = nullptr;
      video_reader_t* reader = nullptr;
      uint32_t fourcc;
      //const VpxInterface *decoder = NULL;

      void open(const fan::string& file_path, properties_t& properties) {
        codec = new vpx_codec_ctx_t;

        reader = vpx_video_reader_open(file_path.c_str());

        if (reader == nullptr) {
          fan::throw_error("failed to open" + file_path);
        }

        vpx_codec_iface_t* face;

        switch (reader->info.codec_fourcc) {
        case VP8_FOURCC: {
          face = vpx_codec_vp8_dx();
          break;
        }
        case VP9_FOURCC: {
          face = vpx_codec_vp9_dx();
          break;
        }
        }

        if (vpx_codec_dec_init(codec, face, nullptr, 0)) {
          fan::throw_error("failed to initialize decoder");
        }

        m_current_delta = 0;
        m_next_delta = 0;
      }
      void close() {
        delete codec;
      }

      void feed_delta(f64_t delta) {
        m_current_delta += delta * 1e+9;
      }

      bool is_time_came() {
        return m_current_delta >= m_next_delta;
      }

      void seek_raw(uint64_t offset) {
        fseek(reader->file, offset, SEEK_SET);
      }

      bool decode_frame(out_t* out) {
        if (!vpx_video_reader_read_frame(reader)) {
          vpx_video_reader_close(reader);
          return false;
        }
        vpx_codec_iter_t iter;
        vpx_image_t* img;
        size_t frame_size;

        uint64_t offset[3]{};

        const unsigned char* frame =
          vpx_video_reader_get_frame(reader, &frame_size);
        if (vpx_codec_decode(codec, frame, (unsigned int)frame_size, NULL, 0)) {
          fan::throw_error("failed to decode frame");
        }

        auto decode_f = [&offset, this](out_t* out, vpx_image_t* img) {
          int plane;

          for (plane = 0; plane < 3; ++plane) {
            const unsigned char* buf = img->planes[plane];
            const int stride = img->stride[plane];
            fan::vec2ui size = vpx_get_plane_size(img, plane);

            size.x *= ((img->fmt & VPX_IMG_FMT_HIGHBITDEPTH) ? 2 : 1);

            if (plane == 0) {
              out->pixel_data.size = fan::vec2(stride, size.y);
            }

            out->pixel_data.linesize[plane] = stride;

            std::memcpy(&out->pixel_data.pixels[plane][offset[plane]], buf, stride * size.y);
            offset[plane] += stride * size.y;

          }
          m_next_delta += 1.0 / reader->info.frame_rate * 1e+9;
        };

        if ((img = vpx_codec_get_frame(codec, &iter)) == nullptr) {
          return true;
        }

        if (out->pixel_data.pixels[0] == nullptr) {
          const int stride = img->stride[0];
          const int h = vpx_get_plane_size(img, 0).y;

          out->pixel_data.pixels[0] = new uint8_t[stride * h];
          out->pixel_data.pixels[1] = new uint8_t[(stride * h) / 2];
          out->pixel_data.pixels[2] = new uint8_t[(stride * h) / 2];
        }

        decode_f(out, img);

        while ((img = vpx_codec_get_frame(codec, &iter)) != nullptr) {
          decode_f(out, img);
        }
        return true;
      }
    };
  }
}
