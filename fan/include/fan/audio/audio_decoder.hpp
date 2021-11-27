#pragma once

#include <fan/io/file.hpp>

#include <string>
#include <stdexcept>

extern "C"
{
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswresample/swresample.h>
};

inline int magic(int n)
{
	int v = n; 

	v--;
	v |= v >> 1;
	v |= v >> 2;
	v |= v >> 4;
	v |= v >> 8;
	v |= v >> 16;
	v++;

	int x = v >> 1;

	return (v - n) > (n - x) ? x : v;
}

namespace fan {

	struct audio_decoder {

		enum class channel_layout_e {
			mono = AV_CH_LAYOUT_MONO,
			stereo = AV_CH_LAYOUT_STEREO
		};

		enum class sample_rates_e {
			rate_8000_hz = 8000,
			rate_11025_hz = 11025,
			rate_16000_hz = 16000,
			rate_25050_hz = 25050,
			rate_44100_hz = 44100,
			rate_48000_hz = 48000,
			rate_88200_hz = 88200,
			rate_176400_hz = 176400,
			rate_192000_hz = 192000,
			rate_352800_hz = 352800,
			rate_384000_hz = 384000
		};

		struct properties_t {

			std::string input;

			channel_layout_e channel_layout = channel_layout_e::mono;
		};

		struct out_t {
			std::string buffer;

			int sample_rate;

			uint64_t duration;
		};

		static out_t decode(properties_t properties)
		{
			out_t out;

			AVFormatContext* format_context = avformat_alloc_context();

			if (avformat_open_input(&format_context, properties.input.c_str(), 0, 0) != 0) {
				throw std::runtime_error("failed to open input stream");
			}
			
			if (avformat_find_stream_info(format_context, 0) < 0) {
				throw std::runtime_error("failed to find stream info");
			}

			// print audio info
			//av_dump_format(format_context, 0, properties.input.c_str(), false);

			int audio_stream = -1;

			for (int i = 0; i < format_context->nb_streams; i++) {
				if (format_context->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {

					audio_stream = i;
					break;
				}
			}

			if (audio_stream == -1) {
				throw std::runtime_error("failed to find audio stream");
			}

			AVCodec* codec = avcodec_find_decoder(format_context->streams[audio_stream]->codecpar->codec_id);
			if (codec == NULL) {
				throw std::runtime_error("failed to find codec");
			}

			AVCodecContext *codec_context = avcodec_alloc_context3(codec);
			if (!codec_context) {
				throw std::runtime_error("out of memory");
			}

			int result = avcodec_parameters_to_context(
				codec_context,
				format_context->streams[audio_stream]->codecpar
			);

			if (result < 0) {
				throw std::runtime_error("failed to set parameters to context");
			}

			if (avcodec_open2(codec_context, codec, NULL) < 0) {
				throw std::runtime_error("failed to open codec");
			}

			AVPacket* packet = (AVPacket*)av_malloc(sizeof(AVPacket));
			av_init_packet(packet);

			uint64_t out_channel_layout = (uint64_t)properties.channel_layout;

			int bits_per_raw_sample = format_context->streams[audio_stream]->codecpar->bits_per_raw_sample;

			if (bits_per_raw_sample == 0) {
				bits_per_raw_sample = format_context->streams[audio_stream]->codecpar->bits_per_coded_sample;
			}

			AVSampleFormat out_sample_fmt = AVSampleFormat::AV_SAMPLE_FMT_S16;

			int out_sample_rate = format_context->streams[audio_stream]->codecpar->sample_rate;//(int)properties.output_rate;

			auto file_size = fan::io::file::file_size(properties.input);

			out.sample_rate = out_sample_rate;

			int out_channels = av_get_channel_layout_nb_channels(out_channel_layout);

			int out_nb_samples = format_context->streams[audio_stream]->codecpar->frame_size;

			// tape fix, approximate nb samples
			if (out_nb_samples == 0) {
				out_nb_samples = bits_per_raw_sample * (format_context->duration * 1e-6 - 1) * out_channels;
				//                     ^ not very accurate              ^ convert microseconds to seconds
			}
			auto x = file_size / (out_channels * out_sample_rate);
			int out_buffer_size = av_samples_get_buffer_size(0, out_channels, out_nb_samples, out_sample_fmt, 0);

			if (out_buffer_size < 0) {
				throw std::runtime_error("failed to get buffer size");
			}

			AVFrame* frame = av_frame_alloc();

			int64_t in_channel_layout = av_get_default_channel_layout(codec_context->channels);

			SwrContext* au_convert_ctx = swr_alloc();
			au_convert_ctx = swr_alloc_set_opts(au_convert_ctx, out_channel_layout, out_sample_fmt, out_sample_rate,
				in_channel_layout, codec_context->sample_fmt, codec_context->sample_rate, 0, 0);

			swr_init(au_convert_ctx);

			int read = 0;

			auto out_buffer = (uint8_t*)av_malloc(out_buffer_size);

			//             conversion from micro to nano
			out.duration = format_context->duration * 1000;

			while (av_read_frame(format_context, packet) >= 0) {
				
				if (packet->stream_index == audio_stream) {
					
					int got_frame;

					int count = avcodec_send_packet(codec_context, packet);

					if (count < 0) {
						throw std::runtime_error("failed to decode audio frame");
					}

					// returns 0 on success
					count = avcodec_receive_frame(codec_context, frame);

					if (count == 0) {
						read += swr_convert(au_convert_ctx, &out_buffer, out_buffer_size, (const uint8_t**)frame->data, frame->nb_samples);

						out.buffer.append(out_buffer, out_buffer + out_buffer_size);
					}
				}

				av_packet_unref(packet);
			}

			swr_free(&au_convert_ctx);

			avcodec_free_context(&codec_context);

			avformat_close_input(&format_context);

			return out;
		}

	};

}
