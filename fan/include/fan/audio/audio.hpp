#pragma once

#include <opus/opusfile.h>

#include <WITCH/TH/TH.h>
#include <WITCH/VAS/VAS1.h>
#include <WITCH/T/T.h>

#include <fan/types/types.hpp>

#define MA_NO_DECODING
#define MA_NO_ENCODING
#define MINIAUDIO_IMPLEMENTATION
#include <fan/audio/miniaudio.h>

#ifdef fan_compiler_visual_studio
	#pragma comment(lib, "lib/opus/opus.lib")
	#pragma comment(lib, "lib/ogg/libogg.lib")
	#pragma comment(lib, "lib/opus/opusfile.lib")
#endif

namespace fan {
	namespace audio {

		struct piece_t;

		namespace{
			namespace constants{
				const uint32_t opus_decode_sample_rate = 48000;

				const f32_t OneSampleTime = (f32_t)1 / opus_decode_sample_rate;

				const uint32_t CallFrameCount = 480;
				const uint32_t ChannelAmount = 2;
				const uint32_t FrameCacheAmount = 4800;
				const uint64_t FrameCacheTime = 1000000000;
			}

			static void opus_print_error(int error) {
				switch (error) {
				case OP_FALSE:  fan::print("OP_FALSE"); break;
				case OP_EOF:  fan::print("OP_EOF"); break;
				case OP_HOLE:  fan::print("OP_HOLE"); break;
				case OP_EREAD:  fan::print("OP_EREAD"); break;
				case OP_EFAULT:  fan::print("OP_EFAULT"); break;
				case OP_EIMPL:  fan::print("OP_EIMPL"); break;
				case OP_EINVAL: fan::print("OP_EINVAL"); break;
				case OP_ENOTFORMAT: fan::print("OP_ENOTFORMAT"); break;
				case OP_EBADHEADER: fan::print("OP_EBADHEADER"); break;
				case OP_EVERSION: fan::print("OP_EVERSION"); break;
				case OP_ENOTAUDIO: fan::print("OP_ENOTAUDIO"); break;
				case OP_EBADPACKET: fan::print("OP_EBADPACKET"); break;
				case OP_EBADLINK: fan::print("OP_EBADLINK"); break;
				case OP_ENOSEEK: fan::print("OP_ENOSEEK"); break;
				case OP_EBADTIMESTAMP: fan::print("OP_EBADTIMESTAMP"); break;
				default:
					fan::print("Unknown error code");
				}
			}

			struct holder_t {
				OggOpusFile *decoder;
				OpusHead *head;
			};

			static void seek(holder_t *holder, uint64_t offset) {
				if (int error = op_pcm_seek(holder->decoder, offset)) {
					opus_print_error(error);
					throw std::runtime_error("failed to seek audio file");
				}
			}

			#define BLL_set_prefix FrameCacheList
			#define BLL_set_type_node uint32_t
			#define BLL_set_node_data \
				uint64_t LastAccessTime; \
				f32_t Frames[constants::FrameCacheAmount][constants::ChannelAmount]; \
				piece_t *piece; \
				uint32_t PieceCacheIndex;
			#include <WITCH/BLL/BLL.h>

			static void decode(holder_t *holder, f32_t *Output, uint64_t offset, uint32_t FrameCount) {
				seek(holder, offset);

				uint64_t TotalRead = 0;

				while (TotalRead != FrameCount) {
					int read = op_read_float_stereo(
						holder->decoder,
						&Output[TotalRead * constants::ChannelAmount],
						(FrameCount - TotalRead) * constants::ChannelAmount);
					if (read <= 0) {
						fan::throw_error("help " + std::to_string(read));
						break;
					}

					TotalRead += read;
				}
			}
		}

		struct audio_t {
			TH_mutex_t play_info_list_mutex;
			VAS1_t play_info_list;

			TH_mutex_t StopQueueMutex;
			VEC_t StopQueue;

			FrameCacheList_t FrameCacheList;

			ma_context context;
			ma_device device;
		};

		namespace{
			struct PieceCache_t{
				FrameCacheList_NodeReference_t ref;
			};
		}
		struct piece_t {
			audio_t *audio;
			holder_t holder;
			uint64_t raw_size;
			PieceCache_t *Cache;
		};

		namespace{
			void GetFrames(audio_t *audio, piece_t *piece, uint64_t Offset, uint64_t Time, f32_t **FramePointer, uint32_t *FrameAmount){
				uint32_t PieceCacheIndex = Offset / constants::FrameCacheAmount;
				PieceCache_t *PieceCache = &piece->Cache[PieceCacheIndex];
				if(PieceCache->ref == (FrameCacheList_NodeReference_t)-1){
					PieceCache->ref = FrameCacheList_NewNodeLast(&audio->FrameCacheList);
					FrameCacheList_Node_t *FrameCacheList_Node = FrameCacheList_GetNodeByReference(&audio->FrameCacheList, PieceCache->ref);
					uint64_t FrameOffset = (uint64_t)PieceCacheIndex * constants::FrameCacheAmount;
					uint32_t FrameAmount = constants::FrameCacheAmount;
					if(FrameOffset + FrameAmount > piece->raw_size){
						FrameAmount = piece->raw_size - FrameOffset;
					}
					decode(&piece->holder, &FrameCacheList_Node->data.Frames[0][0], FrameOffset, FrameAmount);
					FrameCacheList_Node->data.piece = piece;
					FrameCacheList_Node->data.PieceCacheIndex = PieceCacheIndex;
				}
				else{
					FrameCacheList_Reserve(&audio->FrameCacheList, PieceCache->ref);
					FrameCacheList_linkPrev(&audio->FrameCacheList, audio->FrameCacheList.dst, PieceCache->ref);
				}
				FrameCacheList_Node_t *FrameCacheList_Node = FrameCacheList_GetNodeByReference(&audio->FrameCacheList, PieceCache->ref);
				FrameCacheList_Node->data.LastAccessTime = Time;
				*FramePointer = &FrameCacheList_Node->data.Frames[Offset % constants::FrameCacheAmount][0];
				*FrameAmount = constants::FrameCacheAmount - Offset % constants::FrameCacheAmount;
			}
		}

		struct properties_play_t {
			struct{
				uint32_t Loop : 1 = false;
				uint32_t FadeIn : 1 = false;
				uint32_t FadeOut : 1 = false;
			}Flags;
			f32_t FadeFrom;
			f32_t FadeTo;
		};
		struct properties_stop_t{
			f32_t FadeOutTo = 0;
		};
		namespace{
			struct properties_stop_queue_t{
				uint32_t PlayReference;
				properties_stop_t PropertiesStop;
			};
		}

		namespace{
			struct play_info_t {
				piece_t *piece;

				properties_play_t properties;

				uint64_t offset;
			};
		}
		namespace{
			void internal_play_reference_stop(audio_t *audio, uint32_t play_reference, const properties_stop_t *properties_stop){
				if(properties_stop->FadeOutTo != 0){
					play_info_t *play_info = (play_info_t *)VAS1_out(&audio->play_info_list, play_reference);
					if(play_info->properties.Flags.FadeIn){
						play_info->properties.Flags.FadeIn = false;
						play_info->properties.Flags.FadeOut = true;
						f32_t CurrentVolume = play_info->properties.FadeFrom / play_info->properties.FadeTo;
						play_info->properties.FadeTo = properties_stop->FadeOutTo;
						play_info->properties.FadeFrom = ((f32_t)1 - CurrentVolume) * play_info->properties.FadeTo;
					}
					else{
						play_info->properties.FadeFrom = 0;
						play_info->properties.FadeTo = properties_stop->FadeOutTo;
						play_info->properties.Flags.FadeOut = true;
					}
				}
				else{
					TH_lock(&audio->play_info_list_mutex);
					VAS1_unlink(&audio->play_info_list, play_reference);
					TH_unlock(&audio->play_info_list_mutex);
				}
			}
		}
		void play_reference_stop(audio_t *audio, uint32_t PlayReference, const properties_stop_t *PropertiesStop){
			TH_lock(&audio->StopQueueMutex);
			VEC_handle0(&audio->StopQueue, 1);
			properties_stop_queue_t *properties_stop_queue = &((properties_stop_queue_t *)audio->StopQueue.ptr)[audio->StopQueue.Current - 1];
			properties_stop_queue->PlayReference = PlayReference;
			properties_stop_queue->PropertiesStop = *PropertiesStop;
			audio->StopQueue.ptr[audio->StopQueue.Current - 1] = PlayReference;
			TH_unlock(&audio->StopQueueMutex);
		}
		namespace{
			void data_callback(ma_device *Device, void *Output, const void *Input, ma_uint32 FrameCount){
				#ifdef fan_debug
					if(FrameCount != constants::CallFrameCount){
						fan::throw_error("fan_debug");
					}
				#endif
				audio_t *audio = (audio_t *)Device->pUserData;
				uint64_t Time = T_nowi();
				uint32_t LastPlayReference;
				if(audio->StopQueue.Current){
					TH_lock(&audio->play_info_list_mutex);
					TH_lock(&audio->StopQueueMutex);
					for(uint32_t i = 0; i < audio->StopQueue.Current; i++){
						properties_stop_queue_t *properties_stop_queue = &((properties_stop_queue_t *)audio->StopQueue.ptr)[i];
						if(properties_stop_queue->PropertiesStop.FadeOutTo){
							internal_play_reference_stop(audio, properties_stop_queue->PlayReference, &properties_stop_queue->PropertiesStop);
						}
						else{
							VAS1_unlink(&audio->play_info_list, audio->StopQueue.ptr[i]);
						}
					}
					audio->StopQueue.Current = 0;
					TH_unlock(&audio->StopQueueMutex);
					LastPlayReference = audio->play_info_list.dst;
					TH_unlock(&audio->play_info_list_mutex);
				}
				else{
					TH_lock(&audio->play_info_list_mutex);
					LastPlayReference = audio->play_info_list.dst;
					TH_unlock(&audio->play_info_list_mutex);
				}
				uint32_t play_reference = VAS1_GetNodeFirst(&audio->play_info_list);
				while(play_reference != LastPlayReference){
					play_info_t *play_info = (play_info_t *)VAS1_out(&audio->play_info_list, play_reference);
					piece_t *piece = play_info->piece;
					uint32_t OutputIndex = 0;
					uint32_t CanBeReadFrameCount;
					struct{
						f32_t FadePerFrame;
					}CalculatedVariables;
					gt_ReOffset:
					CanBeReadFrameCount = piece->raw_size - play_info->offset;
					if(CanBeReadFrameCount > constants::CallFrameCount){
						CanBeReadFrameCount = constants::CallFrameCount;
					}
					if(play_info->properties.Flags.FadeIn || play_info->properties.Flags.FadeOut){
						f32_t TotalFade = play_info->properties.FadeTo - play_info->properties.FadeFrom;
						f32_t CurrentFadeTime = (f32_t)CanBeReadFrameCount / constants::opus_decode_sample_rate;
						if(TotalFade < CurrentFadeTime){
							CanBeReadFrameCount = TotalFade * constants::opus_decode_sample_rate;
							if(CanBeReadFrameCount == 0){
								uint32_t NextPlayReference = *VAS1_road0(&audio->play_info_list, play_reference);
								properties_stop_t properties_stop;
								internal_play_reference_stop(audio, play_reference, &properties_stop);
								play_reference = NextPlayReference;
								continue;
							}
						}
						CalculatedVariables.FadePerFrame = (f32_t)1 / (play_info->properties.FadeTo * constants::opus_decode_sample_rate);
						if(play_info->properties.Flags.FadeIn){

						}
						else if(play_info->properties.Flags.FadeOut){
							CalculatedVariables.FadePerFrame *= -1;
						}
					}
					while(OutputIndex != CanBeReadFrameCount){
						f32_t *FrameCachePointer;
						uint32_t FrameCacheAmount;
						GetFrames(audio, piece, play_info->offset, Time, &FrameCachePointer, &FrameCacheAmount);
						if(OutputIndex + FrameCacheAmount > CanBeReadFrameCount){
							FrameCacheAmount = CanBeReadFrameCount - OutputIndex;
						}
						if(play_info->properties.Flags.FadeIn || play_info->properties.Flags.FadeOut){
							f32_t CurrentVolume = play_info->properties.FadeFrom / play_info->properties.FadeTo;
							if(play_info->properties.Flags.FadeOut){
								CurrentVolume = (f32_t)1 - CurrentVolume;
							}
							for(uint32_t i = 0; i < FrameCacheAmount; i++){
								for(uint32_t iChannel = 0; iChannel < constants::ChannelAmount; iChannel++){
									((f32_t *)Output)[(OutputIndex + i) * constants::ChannelAmount + iChannel] += FrameCachePointer[i * constants::ChannelAmount + iChannel] * CurrentVolume;
								}
								CurrentVolume += CalculatedVariables.FadePerFrame;
							}
							play_info->properties.FadeFrom += (f32_t)FrameCacheAmount / constants::opus_decode_sample_rate;
						}
						else{
							std::transform(
								&FrameCachePointer[0],
								&FrameCachePointer[FrameCacheAmount * constants::ChannelAmount],
								&((f32_t *)Output)[OutputIndex * constants::ChannelAmount],
								&((f32_t *)Output)[OutputIndex * constants::ChannelAmount],
								std::plus<f32_t>{});
						}
						play_info->offset += FrameCacheAmount;
						OutputIndex += FrameCacheAmount;
					}
					if(play_info->properties.Flags.FadeIn || play_info->properties.Flags.FadeOut){
						if(play_info->properties.Flags.FadeIn){
							if(play_info->properties.FadeFrom + constants::OneSampleTime > play_info->properties.FadeTo){
								play_info->properties.Flags.FadeIn = false;
							}
						}
						else if(play_info->properties.FadeFrom >= play_info->properties.FadeTo){
							uint32_t NextPlayReference = *VAS1_road0(&audio->play_info_list, play_reference);
							properties_stop_t properties_stop;
							internal_play_reference_stop(audio, play_reference, &properties_stop);
							play_reference = NextPlayReference;
							continue;
						}
					}
					if(play_info->offset == piece->raw_size){
						if(play_info->properties.Flags.Loop == true){
							play_info->offset = 0;
							if(OutputIndex != constants::CallFrameCount){
								goto gt_ReOffset;
							}
						}
						uint32_t NextPlayReference = *VAS1_road0(&audio->play_info_list, play_reference);
						properties_stop_t properties_stop;
						internal_play_reference_stop(audio, play_reference, &properties_stop);
						play_reference = NextPlayReference;
					}
					else{
						play_reference = *VAS1_road0(&audio->play_info_list, play_reference);
					}
				}
				for(FrameCacheList_NodeReference_t ref = FrameCacheList_GetNodeFirst(&audio->FrameCacheList); ref != audio->FrameCacheList.dst;){
					FrameCacheList_Node_t *node = FrameCacheList_GetNodeByReference(&audio->FrameCacheList, ref);
					if(node->data.LastAccessTime + constants::FrameCacheTime > Time){
						break;
					}
					node->data.piece->Cache[node->data.PieceCacheIndex].ref = (FrameCacheList_NodeReference_t)-1;
					FrameCacheList_NodeReference_t NextReference = node->NextNodeReference;
					FrameCacheList_unlink(&audio->FrameCacheList, ref);
					ref = NextReference;
				}
			}
		}

		void audio_open(audio_t *audio) {
			TH_mutex_init(&audio->play_info_list_mutex);
			VAS1_open(&audio->play_info_list, sizeof(play_info_t), 0xfff);

			TH_mutex_init(&audio->StopQueueMutex);
			VEC_init(&audio->StopQueue, sizeof(properties_stop_queue_t), A_resize);

			FrameCacheList_open(&audio->FrameCacheList);

			ma_result r;
			if ((r = ma_context_init(NULL, 0, NULL, &audio->context)) != MA_SUCCESS) {
				fan::throw_error("error" + std::to_string(r));
			}

			ma_device_config config = ma_device_config_init(ma_device_type_playback);
			config.playback.format = ma_format_f32;
			config.playback.channels = constants::ChannelAmount;
			config.sampleRate = constants::opus_decode_sample_rate;
			config.dataCallback = data_callback;
			config.pUserData = audio;
			config.periodSizeInFrames = constants::CallFrameCount;

			if((r = ma_device_init(&audio->context, &config, &audio->device)) != MA_SUCCESS) {
				fan::throw_error("ma_device_init" + std::to_string(r));
			}
			if((r = ma_device_start(&audio->device)) != MA_SUCCESS){
				fan::throw_error("ma_device_start" + std::to_string(r));
			}
		}

		void audio_close(audio_t *audio) {
			fan::throw_error("this part needs more attention");
			VAS1_close(&audio->play_info_list);
			FrameCacheList_close(&audio->FrameCacheList);
			ma_context_uninit(&audio->context);
		}

		void audio_set_volume(audio_t *audio, f32_t Volume){
			if(ma_device_set_master_volume(&audio->device, Volume) != MA_SUCCESS){
				fan::throw_error("audio_set_volume");
			}
		}
		f32_t audio_get_volume(audio_t *audio){
			f32_t Volume;
			if(ma_device_get_master_volume(&audio->device, &Volume) != MA_SUCCESS){
				fan::throw_error("audio_get_volume");
			}
			return Volume;
		}

		uint32_t piece_play(audio_t *audio, piece_t *piece, const properties_play_t *properties) {
			TH_lock(&audio->play_info_list_mutex);
			uint32_t play_reference = VAS1_NewNodeLast(&audio->play_info_list);
			play_info_t *play_info = (play_info_t *)VAS1_out(&audio->play_info_list, play_reference);
			play_info->piece = piece;
			play_info->properties = *properties;
			play_info->offset = 0;
			TH_unlock(&audio->play_info_list_mutex);

			return play_reference;
		}

		namespace{
			static holder_t get_file_info(const std::string& path) {

				holder_t holder;

				int error = 0;
				holder.decoder = op_open_file(path.c_str(), &error);
				if (error != 0) {
					opus_print_error(error);
					throw std::runtime_error("failed to open opus file");
				}

				holder.head = (OpusHead *)op_head(holder.decoder, 0);

				return holder;
			}
		}
		void piece_open_from_file(fan::audio::audio_t *audio, fan::audio::piece_t *piece, const std::string& path) {
			piece->audio = audio;
			piece->holder = get_file_info(path);
			piece->raw_size = op_pcm_total(piece->holder.decoder, 0);
			uint32_t CacheAmount = piece->raw_size / constants::FrameCacheAmount + !!(piece->raw_size % constants::FrameCacheAmount);
			piece->Cache = (PieceCache_t *)A_resize(0, CacheAmount * sizeof(PieceCache_t));
			for(uint32_t i = 0; i < CacheAmount; i++){
				piece->Cache[i].ref = (FrameCacheList_NodeReference_t)-1;
			}
		}

	}
}
