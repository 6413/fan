#pragma once

#include <fan/math/vector.hpp>

#define __STDC_CONSTANT_MACROS

extern "C"
{
	#include <libavutil/imgutils.h>
	#include <libavcodec/avcodec.h>

	#include <libavutil/common.h>

	#include <libavutil/mathematics.h>
	#include <libavformat/avformat.h>

	#include <libswscale/swscale.h>
}

namespace fan {

	namespace image_loader {

		struct image_data {
			image_data() : data{}, linesize{} {}
			uint8_t* data[4];
			int linesize[4];
			fan::vec2i size;
			AVPixelFormat format;
		};

		static uint_t get_stride_multiplier(AVPixelFormat format) {
			switch (format) {
				case AVPixelFormat::AV_PIX_FMT_BGR24:
				case AVPixelFormat::AV_PIX_FMT_RGB24:
				{
					return 3;
				}
				case AVPixelFormat::AV_PIX_FMT_BGRA:
				{
					return 4;
				}
				default: {
					return -1;
				}
			}
		}

		static fan::image_loader::image_data convert_format(const fan::image_loader::image_data& image_data, AVPixelFormat new_format) {

			image_loader::image_data new_data;

			auto context = sws_getContext(
				image_data.size.x, 
				image_data.size.y, 
				image_data.format, 
				image_data.size.x, 
				image_data.size.y, 
				new_format,
				SWS_BILINEAR,
				0, 
				0, 
				0
			);

			if (!context) {
				fan::print("failed to get context");
				return {};
			}

			new_data.linesize[0] = get_stride_multiplier(new_format) * image_data.size.x;

			if (av_image_alloc(
				new_data.data,
				new_data.linesize,
				image_data.size.x,
				image_data.size.y,
				new_format,
				2
				) < 0) {
				fan::print("failed to allocate image");
				return {};
			}

			sws_scale(
				context, 
				image_data.data, 
				image_data.linesize,
				0, 
				image_data.size.y, 
				new_data.data,
				new_data.linesize
			);

			sws_freeContext(context);

			av_freep((void*)&image_data.data);

			new_data.format = new_format;
			new_data.size = image_data.size;

			return new_data;
		}

		static fan::image_loader::image_data load_image(const std::string& filename) {

			fan::image_loader::image_data image_data;

			AVInputFormat *iformat = NULL;
			AVFormatContext *format_ctx = NULL;
			const AVCodec *codec;
			AVCodecContext *codec_ctx = NULL;
			AVCodecParameters *par;
			AVFrame *frame = NULL;
			AVPacket pkt;
			AVDictionary *opt=NULL;

			iformat = av_find_input_format("image2pipe");
			if (avformat_open_input(&format_ctx, filename.c_str(), iformat, 0) < 0) {
				fan::print("failed to open input file: ", filename);
				return {};
			}

			if (avformat_find_stream_info(format_ctx, 0) < 0) {
				fan::print("find stream info failed");
				goto end;
			}

			par = format_ctx->streams[0]->codecpar;
			codec = avcodec_find_decoder(par->codec_id);

			if (!codec) {
				fan::print("failed to find codec");
				goto end;
			}

			codec_ctx = avcodec_alloc_context3(codec);
			if (!codec_ctx) {
				fan::print("failed to alloc video decoder context");
				goto end;
			}

			if (avcodec_parameters_to_context(codec_ctx, par) < 0) {
				fan::print("failed to copy codec parameters to decoder context");
				goto end;
			}

			av_dict_set(&opt, "thread_type", "slice", 0);

			if (avcodec_open2(codec_ctx, codec, &opt) < 0) {
				fan::print("failed to open codec");
				goto end;
			}

			if (!(frame = av_frame_alloc()) ) {
				fan::print("failed to alloc frame");
				goto end;
			}

			if (av_read_frame(format_ctx, &pkt) < 0) {
				fan::print("failed to read frame from file");
				goto end;
			}

			if (avcodec_send_packet(codec_ctx, &pkt) < 0) {
				av_packet_unref(&pkt);
				fan::print("error submitting a packet to decoder");
				goto end;
			}

			av_packet_unref(&pkt);

			if (avcodec_receive_frame(codec_ctx, frame) < 0) {
				fan::print("failed to decode image from file");
				goto end;
			}

			image_data.size.x = frame->width;
			image_data.size.y = frame->height;

			image_data.format = (AVPixelFormat)frame->format;

			if (av_image_alloc(
				image_data.data,
				image_data.linesize,
				image_data.size.x,
				image_data.size.y,
				image_data.format,
				16) < 0) {
				fan::print("failed to allocate image");
				goto end;
			}

			av_image_copy(
				image_data.data, 
				image_data.linesize, 
				(const uint8_t **)frame->data, 
				frame->linesize, 
				image_data.format, 
				image_data.size.x, 
				image_data.size.y
			);

end:

			switch (image_data.format) {
				
				case AV_PIX_FMT_YUVJ420P:
				{
					image_data.format = AV_PIX_FMT_YUV420P;
					image_data = convert_format(image_data, AVPixelFormat::AV_PIX_FMT_RGB24);
					break;
				}
				case AV_PIX_FMT_YUVJ422P:
				{
					image_data.format = AV_PIX_FMT_YUV422P;
					image_data = convert_format(image_data, AVPixelFormat::AV_PIX_FMT_RGB24);
					break;
				}
				case AV_PIX_FMT_YUVJ444P:
				{
					image_data.format = AV_PIX_FMT_YUV444P;
					image_data = convert_format(image_data, AVPixelFormat::AV_PIX_FMT_RGB24);
					break;
				}
				case AV_PIX_FMT_YUVJ440P:
				{
					image_data.format = AV_PIX_FMT_YUV440P;
					image_data = convert_format(image_data, AVPixelFormat::AV_PIX_FMT_RGB24);
					break;
				}
			}

			avcodec_free_context(&codec_ctx);
			avformat_close_input(&format_ctx);
			av_frame_free(&frame);
			av_dict_free(&opt);

			return image_data;
		}

	}

}