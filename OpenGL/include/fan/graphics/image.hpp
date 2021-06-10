#pragma once

#include <fan/types/vector.hpp>

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

#ifdef FAN_PLATFORM_WINDOWS

	#pragma comment(lib, "lib/av/avcodec.lib")
	#pragma comment(lib, "lib/av/avdevice.lib")
	#pragma comment(lib, "lib/av/avfilter.lib")
	#pragma comment(lib, "lib/av/avformat.lib")
	#pragma comment(lib, "lib/av/avutil.lib")
	#pragma comment(lib, "lib/av/postproc.lib")
	#pragma comment(lib, "lib/av/swresample.lib")
	#pragma comment(lib, "lib/av/swscale.lib")

#endif

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
				case AVPixelFormat::AV_PIX_FMT_PAL8:
				{
					return 1;
				}
				case AVPixelFormat::AV_PIX_FMT_BGR24:
				case AVPixelFormat::AV_PIX_FMT_RGB24:
				{
					return 3;
				}
				case AVPixelFormat::AV_PIX_FMT_RGBA:
				case AVPixelFormat::AV_PIX_FMT_BGRA:
				{
					return 4;
				}
				default: {
					return -1;
				}
			}
		}

		static long int crv_tab[256];   
		static long int cbu_tab[256];   
		static long int cgu_tab[256];   
		static long int cgv_tab[256];   
		static long int tab_76309[256]; 
		static unsigned char clp[1024];  

		void init_yuv420p_table() 
		{   
			long int crv,cbu,cgu,cgv;   
			int i,ind;      

			crv = 104597; cbu = 132201;  
			cgu = 25675;  cgv = 53279;   

			for (i = 0; i < 256; i++)    
			{   
				crv_tab[i] = (i-128) * crv;   
				cbu_tab[i] = (i-128) * cbu;   
				cgu_tab[i] = (i-128) * cgu;   
				cgv_tab[i] = (i-128) * cgv;   
				tab_76309[i] = 76309*(i-16);   
			}   

			for (i = 0; i < 384; i++)   
				clp[i] = 0;   
			ind = 384;   
			for (i = 0;i < 256; i++)   
				clp[ind++] = i;   
			ind = 640;   
			for (i = 0;i < 384; i++)   
				clp[ind++] = 255;   
		}

		static void yuv420p_to_rgb24(uint8_t * const * yuvbuffer, uint8_t* rgbbuffer, int width, int height)
		{
			int y1, y2, u, v;    
			uint8_t *py1, *py2;   
			int i, j, c1, c2, c3, c4;   
			uint8_t *d1, *d2;   
			uint8_t *src_u, *src_v;
			static int init_yuv420p = 0;

			src_u = yuvbuffer[1];   // u
			src_v = yuvbuffer[2];  // v

			py1 = yuvbuffer[0];   // y
			py2 = py1 + width;   
			d1 = rgbbuffer;   
			d2 = d1 + 3 * width;   

			if (init_yuv420p == 0)
			{
				init_yuv420p_table();
				init_yuv420p = 1;
			}

			for (j = 0; j < height; j += 2)    
			{    
				for (i = 0; i < width; i += 2)    
				{
					u = *src_u++;   
					v = *src_v++;   

					c1 = crv_tab[v];   
					c2 = cgu_tab[u];   
					c3 = cgv_tab[v];   
					c4 = cbu_tab[u];   

					y1 = tab_76309[*py1++];    
					*d1++ = clp[384+((y1 + c1)>>16)];     
					*d1++ = clp[384+((y1 - c2 - c3)>>16)];   
					*d1++ = clp[384+((y1 + c4)>>16)];   

					y2 = tab_76309[*py2++];   
					*d2++ = clp[384+((y2 + c1)>>16)];     
					*d2++ = clp[384+((y2 - c2 - c3)>>16)];   
					*d2++ = clp[384+((y2 + c4)>>16)];   

					y1 = tab_76309[*py1++];   
					*d1++ = clp[384+((y1 + c1)>>16)];     
					*d1++ = clp[384+((y1 - c2 - c3)>>16)];   
					*d1++ = clp[384+((y1 + c4)>>16)];   

					y2 = tab_76309[*py2++];   
					*d2++ = clp[384+((y2 + c1)>>16)];     
					*d2++ = clp[384+((y2 - c2 - c3)>>16)];   
					*d2++ = clp[384+((y2 + c4)>>16)];   
				}
				d1  += 3*width;
				d2  += 3*width;
				py1 += width;
				py2 += width;
			}
		}

		static fan::image_loader::image_data convert_format(const fan::image_loader::image_data& image_data, AVPixelFormat new_format) {

			image_loader::image_data new_data;

			new_data.data[0] = new uint8_t[image_data.size.x * image_data.size.y * 3];

			auto size = image_data.size.x * image_data.size.y;

			//yuv420p_to_rgb24(image_data.data, new_data.data[0], image_data.size.x, image_data.size.y);

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

			//if (av_image_alloc(
			//	new_data.data,
			//	new_data.linesize,
			//	image_data.size.x,
			//	image_data.size.y,
			//	new_format,
			//	1
			//	) < 0) {
			//	fan::print("failed to allocate image");
			//	return {};
			//}

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

			//av_freep((void*)&image_data.data);

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

			/*image_data.linesize[0] = get_stride_multiplier(image_data.format) * image_data.size.x;

			image_data.data[0] = new uint8_t[image_data.linesize[0] * image_data.size.y];
			image_data.data[1] = new uint8_t[image_data.linesize[0] * image_data.size.y];*/

			if (av_image_alloc(
				image_data.data,
				image_data.linesize,
				image_data.size.x,
				image_data.size.y,
				image_data.format,
				1) < 0) {
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
				case AV_PIX_FMT_RGBA:
				case AV_PIX_FMT_RGB24:
				{
					break;
				}
				default:
				{
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