#pragma once

#include <opus/opusfile.h>

#include <WITCH/TH/TH.h>
#include <WITCH/T/T.h>

#include <fan/types/types.h>

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

		namespace {
			namespace constants {
				const uint32_t opus_decode_sample_rate = 48000;

				const f32_t OneSampleTime = (f32_t)1 / opus_decode_sample_rate;

				const uint32_t CallFrameCount = 480;
				const uint32_t ChannelAmount = 2;
				const uint32_t FrameCacheAmount = 4800;
				const uint64_t FrameCacheTime = opus_decode_sample_rate / CallFrameCount * 1; /* 1 second */
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
				OggOpusFile* decoder;
				OpusHead* head;
			};

			static void seek(holder_t* holder, uint64_t offset) {
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
#define BLL_set_ResizeListAfterClear 1
#include <WITCH/BLL/BLL.h>

			static void decode(holder_t* holder, f32_t* Output, uint64_t offset, uint32_t FrameCount) {
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

		struct PropertiesSoundPlay_t {
			struct {
				uint32_t Loop : 1 = false;
				uint32_t FadeIn : 1 = false;
				uint32_t FadeOut : 1 = false;
			}Flags;
			f32_t FadeFrom;
			f32_t FadeTo;
		};
		struct PropertiesSoundStop_t {
			f32_t FadeOutTo = 0;
		};
		namespace {
#define BLL_set_prefix PlayInfoList
#define BLL_set_type_node uint32_t
#define BLL_set_node_data \
				piece_t *piece; \
				uint32_t GroupID; \
				uint32_t PlayID; \
				PropertiesSoundPlay_t properties; \
				uint64_t offset;
#define BLL_set_ResizeListAfterClear 1
#include <WITCH/BLL/BLL.h>

			enum class MessageType_t {
				SoundPlay,
				SoundStop,
				PauseGroup,
				ResumeGroup,
				StopGroup
			};
			struct Message_t {
				MessageType_t Type;
				union {
					struct {
						uint32_t PlayInfoReference;
					}SoundPlay;
					struct {
						uint32_t PlayInfoReference;
						PropertiesSoundStop_t Properties;
					}SoundStop;
					struct {
						uint32_t GroupID;
					}PauseGroup;
					struct {
						uint32_t GroupID;
					}ResumeGroup;
					struct {
						uint32_t GroupID;
					}StopGroup;
				}Data;
			};
		}

		namespace {
			struct Group_t {
				PlayInfoList_NodeReference_t FirstReference;
				PlayInfoList_NodeReference_t LastReference;
			};
			struct Play_t {
				PlayInfoList_NodeReference_t Reference;
			};
		}
		struct audio_t {
			TH_mutex_t PlayInfoListMutex;
			PlayInfoList_t PlayInfoList;

			uint32_t GroupAmount;
			Group_t* GroupList;

			VEC_t PlayList;

			TH_mutex_t MessageQueueListMutex;
			VEC_t MessageQueueList;

			FrameCacheList_t FrameCacheList;

			uint64_t Tick;

			ma_context context;
			ma_device device;
		};

		namespace {
			struct PieceCache_t {
				FrameCacheList_NodeReference_t ref;
			};
		}
		struct piece_t {
			audio_t* audio;
			holder_t holder;
			uint64_t raw_size;
			PieceCache_t* Cache;
		};

		namespace {
			void GetFrames(audio_t* audio, piece_t* piece, uint64_t Offset, uint64_t Time, f32_t** FramePointer, uint32_t* FrameAmount) {
				uint32_t PieceCacheIndex = Offset / constants::FrameCacheAmount;
				PieceCache_t* PieceCache = &piece->Cache[PieceCacheIndex];
				if (PieceCache->ref == (FrameCacheList_NodeReference_t)-1) {
					PieceCache->ref = FrameCacheList_NewNodeLast(&audio->FrameCacheList);
					FrameCacheList_Node_t* FrameCacheList_Node = FrameCacheList_GetNodeByReference(&audio->FrameCacheList, PieceCache->ref);
					uint64_t FrameOffset = (uint64_t)PieceCacheIndex * constants::FrameCacheAmount;
					uint32_t FrameAmount = constants::FrameCacheAmount;
					if (FrameOffset + FrameAmount > piece->raw_size) {
						FrameAmount = piece->raw_size - FrameOffset;
					}
					decode(&piece->holder, &FrameCacheList_Node->data.Frames[0][0], FrameOffset, FrameAmount);
					FrameCacheList_Node->data.piece = piece;
					FrameCacheList_Node->data.PieceCacheIndex = PieceCacheIndex;
				}
				else {
					FrameCacheList_Unlink(&audio->FrameCacheList, PieceCache->ref);
					FrameCacheList_linkPrev(&audio->FrameCacheList, audio->FrameCacheList.dst, PieceCache->ref);
				}
				FrameCacheList_Node_t* FrameCacheList_Node = FrameCacheList_GetNodeByReference(&audio->FrameCacheList, PieceCache->ref);
				FrameCacheList_Node->data.LastAccessTime = Time;
				*FramePointer = &FrameCacheList_Node->data.Frames[Offset % constants::FrameCacheAmount][0];
				*FrameAmount = constants::FrameCacheAmount - Offset % constants::FrameCacheAmount;
			}
		}

		namespace {
			void RemoveFromPlayList(audio_t* audio, uint32_t PlayID) {
				/* super fast remove */
				((Play_t*)audio->PlayList.ptr)[PlayID] = ((Play_t*)audio->PlayList.ptr)[--audio->PlayList.Current];

				/* moved one needs update */
				PlayInfoList_NodeReference_t PlayInfoReference = ((Play_t*)audio->PlayList.ptr)[PlayID].Reference;
				PlayInfoList_Node_t* PlayInfoNode = PlayInfoList_GetNodeByReference(&audio->PlayInfoList, PlayInfoReference);
				PlayInfoNode->data.PlayID = PlayID;
			}
			void RemoveFromPlayInfoList(audio_t* audio, uint32_t PlayInfoReference, const PropertiesSoundStop_t* Properties) {
				PlayInfoList_Node_t* PlayInfoNode = PlayInfoList_GetNodeByReference(&audio->PlayInfoList, PlayInfoReference);
				if (PlayInfoNode->data.PlayID == (uint32_t)-1) {
					/* properties are ignored */
					TH_lock(&audio->PlayInfoListMutex);
					PlayInfoList_Unlink(&audio->PlayInfoList, PlayInfoReference);
					PlayInfoList_Recycle(&audio->PlayInfoList, PlayInfoReference);
					TH_unlock(&audio->PlayInfoListMutex);
				}
				else {
					PlayInfoList_Node_t* PlayInfoNode = PlayInfoList_GetNodeByReference(&audio->PlayInfoList, PlayInfoReference);
					if (Properties->FadeOutTo != 0) {
						PropertiesSoundPlay_t* PropertiesPlay = &PlayInfoNode->data.properties;
						if (PropertiesPlay->Flags.FadeIn) {
							PropertiesPlay->Flags.FadeIn = false;
							PropertiesPlay->Flags.FadeOut = true;
							f32_t CurrentVolume = PropertiesPlay->FadeFrom / PropertiesPlay->FadeTo;
							PropertiesPlay->FadeTo = Properties->FadeOutTo;
							PropertiesPlay->FadeFrom = ((f32_t)1 - CurrentVolume) * PropertiesPlay->FadeTo;
						}
						else {
							PropertiesPlay->FadeFrom = 0;
							PropertiesPlay->FadeTo = Properties->FadeOutTo;
							PropertiesPlay->Flags.FadeOut = true;
						}
					}
					else {
						RemoveFromPlayList(audio, PlayInfoNode->data.PlayID);
						TH_lock(&audio->PlayInfoListMutex);
						PlayInfoList_Unlink(&audio->PlayInfoList, PlayInfoReference);
						PlayInfoList_Recycle(&audio->PlayInfoList, PlayInfoReference);
						TH_unlock(&audio->PlayInfoListMutex);
					}
				}
			}
		}
		void SoundStop(audio_t* audio, PlayInfoList_NodeReference_t PlayInfoReference, const PropertiesSoundStop_t* Properties) {
			TH_lock(&audio->MessageQueueListMutex);
			VEC_handle0(&audio->MessageQueueList, 1);
			Message_t* Message = &((Message_t*)audio->MessageQueueList.ptr)[audio->MessageQueueList.Current - 1];
			Message->Type = MessageType_t::SoundStop;
			Message->Data.SoundStop.PlayInfoReference = PlayInfoReference;
			Message->Data.SoundStop.Properties = *Properties;
			TH_unlock(&audio->MessageQueueListMutex);
		}
		namespace {
			void AddSoundToPlay(audio_t* audio, PlayInfoList_NodeReference_t PlayInfoReference) {
				VEC_handle0(&audio->PlayList, 1);
				uint32_t PlayID = audio->PlayList.Current - 1;
				Play_t* Play = &((Play_t*)audio->PlayList.ptr)[PlayID];
				Play->Reference = PlayInfoReference;
				PlayInfoList_Node_t* PlayInfoNode = PlayInfoList_GetNodeByReference(&audio->PlayInfoList, PlayInfoReference);
#if fan_debug
				if (PlayInfoNode->data.PlayID != (uint32_t)-1) {
					/* trying play sound that already playing */
					fan::throw_error("fan_debug");
				}
#endif
				PlayInfoNode->data.PlayID = PlayID;
			}
			void data_callback(ma_device* Device, void* Output, const void* Input, ma_uint32 FrameCount) {
#if fan_debug
				if (FrameCount != constants::CallFrameCount) {
					fan::throw_error("fan_debug");
				}
#endif
				audio_t* audio = (audio_t*)Device->pUserData;
				if (audio->MessageQueueList.Current) {
					TH_lock(&audio->MessageQueueListMutex);
					for (uint32_t i = 0; i < audio->MessageQueueList.Current; i++) {
						Message_t* Message = &((Message_t*)audio->MessageQueueList.ptr)[i];
						switch (Message->Type) {
						case MessageType_t::SoundPlay: {
							AddSoundToPlay(audio, Message->Data.SoundPlay.PlayInfoReference);
							break;
						}
						case MessageType_t::SoundStop: {
							RemoveFromPlayInfoList(audio, Message->Data.SoundStop.PlayInfoReference, &Message->Data.SoundStop.Properties);
							break;
						}
						case MessageType_t::PauseGroup: {
							uint32_t GroupID = Message->Data.PauseGroup.GroupID;
							PlayInfoList_NodeReference_t LastPlayInfoReference = audio->GroupList[GroupID].LastReference;
							PlayInfoList_NodeReference_t PlayInfoReference = audio->GroupList[GroupID].FirstReference;
							TH_lock(&audio->PlayInfoListMutex);
							PlayInfoReference = PlayInfoList_GetNodeByReference(&audio->PlayInfoList, PlayInfoReference)->NextNodeReference;
							TH_unlock(&audio->PlayInfoListMutex);
							while (PlayInfoReference != LastPlayInfoReference) {
								PlayInfoList_Node_t* PlayInfoNode = PlayInfoList_GetNodeByReference(&audio->PlayInfoList, PlayInfoReference);
								if (PlayInfoNode->data.PlayID != (uint32_t)-1) {
									RemoveFromPlayList(audio, PlayInfoNode->data.PlayID);
									PlayInfoNode->data.PlayID = (uint32_t)-1;
								}
								PlayInfoReference = PlayInfoNode->NextNodeReference;
							}
							break;
						}
						case MessageType_t::ResumeGroup: {
							uint32_t GroupID = Message->Data.PauseGroup.GroupID;
							PlayInfoList_NodeReference_t LastPlayInfoReference = audio->GroupList[GroupID].LastReference;
							PlayInfoList_NodeReference_t PlayInfoReference = audio->GroupList[GroupID].FirstReference;
							TH_lock(&audio->PlayInfoListMutex);
							PlayInfoReference = PlayInfoList_GetNodeByReference(&audio->PlayInfoList, PlayInfoReference)->NextNodeReference;
							TH_unlock(&audio->PlayInfoListMutex);
							while (PlayInfoReference != LastPlayInfoReference) {
								PlayInfoList_Node_t* PlayInfoNode = PlayInfoList_GetNodeByReference(&audio->PlayInfoList, PlayInfoReference);
								if (PlayInfoNode->data.PlayID == (uint32_t)-1) {
									AddSoundToPlay(audio, PlayInfoReference);
								}
								PlayInfoReference = PlayInfoNode->NextNodeReference;
							}
							break;
						}
						case MessageType_t::StopGroup: {
							uint32_t GroupID = Message->Data.PauseGroup.GroupID;
							PlayInfoList_NodeReference_t LastPlayInfoReference = audio->GroupList[GroupID].LastReference;
							PlayInfoList_NodeReference_t PlayInfoReference = audio->GroupList[GroupID].FirstReference;
							TH_lock(&audio->PlayInfoListMutex);
							PlayInfoReference = PlayInfoList_GetNodeByReference(&audio->PlayInfoList, PlayInfoReference)->NextNodeReference;
							while (PlayInfoReference != LastPlayInfoReference) {
								PlayInfoList_Node_t* PlayInfoNode = PlayInfoList_GetNodeByReference(&audio->PlayInfoList, PlayInfoReference);
								PropertiesSoundStop_t Properties;
								RemoveFromPlayInfoList(audio, PlayInfoReference, &Properties);
								PlayInfoReference = PlayInfoNode->NextNodeReference;
							}
							TH_unlock(&audio->PlayInfoListMutex);
						}
						}
					}
					audio->MessageQueueList.Current = 0;
					TH_unlock(&audio->MessageQueueListMutex);
				}
				for (uint32_t PlayID = 0; PlayID < audio->PlayList.Current;) {
					Play_t* Play = &((Play_t*)audio->PlayList.ptr)[PlayID];
					PlayInfoList_NodeReference_t PlayInfoReference = Play->Reference;
					PlayInfoList_Node_t* PlayInfoNode = PlayInfoList_GetNodeByReference(&audio->PlayInfoList, PlayInfoReference);
					piece_t* piece = PlayInfoNode->data.piece;
					PropertiesSoundPlay_t* Properties = &PlayInfoNode->data.properties;
					uint32_t OutputIndex = 0;
					uint32_t CanBeReadFrameCount;
					struct {
						f32_t FadePerFrame;
					}CalculatedVariables;
				gt_ReOffset:
					CanBeReadFrameCount = piece->raw_size - PlayInfoNode->data.offset;
					if (CanBeReadFrameCount > constants::CallFrameCount) {
						CanBeReadFrameCount = constants::CallFrameCount;
					}
					if (Properties->Flags.FadeIn || Properties->Flags.FadeOut) {
						f32_t TotalFade = Properties->FadeTo - Properties->FadeFrom;
						f32_t CurrentFadeTime = (f32_t)CanBeReadFrameCount / constants::opus_decode_sample_rate;
						if (TotalFade < CurrentFadeTime) {
							CanBeReadFrameCount = TotalFade * constants::opus_decode_sample_rate;
							if (CanBeReadFrameCount == 0) {
								PropertiesSoundStop_t PropertiesSoundStop;
								RemoveFromPlayInfoList(audio, PlayInfoReference, &PropertiesSoundStop);
								continue;
							}
						}
						CalculatedVariables.FadePerFrame = (f32_t)1 / (Properties->FadeTo * constants::opus_decode_sample_rate);
						if (Properties->Flags.FadeIn) {

						}
						else if (Properties->Flags.FadeOut) {
							CalculatedVariables.FadePerFrame *= -1;
						}
					}
					while (OutputIndex != CanBeReadFrameCount) {
						f32_t* FrameCachePointer;
						uint32_t FrameCacheAmount;
						GetFrames(audio, piece, PlayInfoNode->data.offset, audio->Tick, &FrameCachePointer, &FrameCacheAmount);
						if (OutputIndex + FrameCacheAmount > CanBeReadFrameCount) {
							FrameCacheAmount = CanBeReadFrameCount - OutputIndex;
						}
						if (Properties->Flags.FadeIn || Properties->Flags.FadeOut) {
							f32_t CurrentVolume = Properties->FadeFrom / Properties->FadeTo;
							if (Properties->Flags.FadeOut) {
								CurrentVolume = (f32_t)1 - CurrentVolume;
							}
							for (uint32_t i = 0; i < FrameCacheAmount; i++) {
								for (uint32_t iChannel = 0; iChannel < constants::ChannelAmount; iChannel++) {
									((f32_t*)Output)[(OutputIndex + i) * constants::ChannelAmount + iChannel] += FrameCachePointer[i * constants::ChannelAmount + iChannel] * CurrentVolume;
								}
								CurrentVolume += CalculatedVariables.FadePerFrame;
							}
							Properties->FadeFrom += (f32_t)FrameCacheAmount / constants::opus_decode_sample_rate;
						}
						else {
							std::transform(
								&FrameCachePointer[0],
								&FrameCachePointer[FrameCacheAmount * constants::ChannelAmount],
								&((f32_t*)Output)[OutputIndex * constants::ChannelAmount],
								&((f32_t*)Output)[OutputIndex * constants::ChannelAmount],
								std::plus<f32_t>{});
						}
						PlayInfoNode->data.offset += FrameCacheAmount;
						OutputIndex += FrameCacheAmount;
					}
					if (Properties->Flags.FadeIn || Properties->Flags.FadeOut) {
						if (Properties->Flags.FadeIn) {
							if (Properties->FadeFrom + constants::OneSampleTime > Properties->FadeTo) {
								Properties->Flags.FadeIn = false;
							}
						}
						else if (Properties->FadeFrom >= Properties->FadeTo) {
							PropertiesSoundStop_t PropertiesSoundStop;
							RemoveFromPlayInfoList(audio, PlayInfoReference, &PropertiesSoundStop);
							continue;
						}
					}
					if (PlayInfoNode->data.offset == piece->raw_size) {
						if (Properties->Flags.Loop == true) {
							PlayInfoNode->data.offset = 0;
							if (OutputIndex != constants::CallFrameCount) {
								goto gt_ReOffset;
							}
						}
						PropertiesSoundStop_t PropertiesSoundStop;
						RemoveFromPlayInfoList(audio, PlayInfoReference, &PropertiesSoundStop);
						continue;
					}
					PlayID++;
				}
				for (FrameCacheList_NodeReference_t ref = FrameCacheList_GetNodeFirst(&audio->FrameCacheList); ref != audio->FrameCacheList.dst;) {
					FrameCacheList_Node_t* node = FrameCacheList_GetNodeByReference(&audio->FrameCacheList, ref);
					if (node->data.LastAccessTime + constants::FrameCacheTime > audio->Tick) {
						break;
					}
					node->data.piece->Cache[node->data.PieceCacheIndex].ref = (FrameCacheList_NodeReference_t)-1;
					FrameCacheList_NodeReference_t NextReference = node->NextNodeReference;
					FrameCacheList_Unlink(&audio->FrameCacheList, ref);
					FrameCacheList_Recycle(&audio->FrameCacheList, ref);
					ref = NextReference;
				}
				audio->Tick++;
			}
		}

		void audio_close(audio_t* audio) {
			ma_device_uninit(&audio->device);
			ma_context_uninit(&audio->context);
			PlayInfoList_close(&audio->PlayInfoList);
			A_resize(audio->GroupList, 0);
			VEC_free(&audio->PlayList);
			VEC_free(&audio->MessageQueueList);
			FrameCacheList_close(&audio->FrameCacheList);
		}
		void audio_open(audio_t* audio, uint32_t GroupAmount) {
			TH_mutex_init(&audio->PlayInfoListMutex);
			PlayInfoList_open(&audio->PlayInfoList);

			audio->GroupAmount = GroupAmount;
			audio->GroupList = (Group_t*)A_resize(0, sizeof(Group_t) * audio->GroupAmount);
			for (uint32_t i = 0; i < audio->GroupAmount; i++) {
				audio->GroupList[i].FirstReference = PlayInfoList_NewNodeLast_alloc(&audio->PlayInfoList);
				audio->GroupList[i].LastReference = PlayInfoList_NewNodeLast_alloc(&audio->PlayInfoList);
			}

			VEC_init(&audio->PlayList, sizeof(Play_t), A_resize);

			TH_mutex_init(&audio->MessageQueueListMutex);
			VEC_init(&audio->MessageQueueList, sizeof(Message_t), A_resize);

			audio->Tick = 0;

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

			if ((r = ma_device_init(&audio->context, &config, &audio->device)) != MA_SUCCESS) {
				fan::throw_error("ma_device_init" + std::to_string(r));
			}
		}

		void audio_stop(audio_t* audio) {
			if (ma_device_stop(&audio->device) != MA_SUCCESS) {
				fan::throw_error("audio_stop");
			}
		}
		void audio_start(audio_t* audio) {
			if (ma_device_start(&audio->device) != MA_SUCCESS) {
				fan::throw_error("audio_start");
			}
		}

		void audio_pause_group(audio_t* audio, uint32_t GroupID) {
			TH_lock(&audio->MessageQueueListMutex);
			VEC_handle0(&audio->MessageQueueList, 1);
			Message_t* Message = &((Message_t*)audio->MessageQueueList.ptr)[audio->MessageQueueList.Current - 1];
			Message->Type = MessageType_t::PauseGroup;
			Message->Data.PauseGroup.GroupID = GroupID;
			TH_unlock(&audio->MessageQueueListMutex);
		}
		void audio_resume_group(audio_t* audio, uint32_t GroupID) {
			TH_lock(&audio->MessageQueueListMutex);
			VEC_handle0(&audio->MessageQueueList, 1);
			Message_t* Message = &((Message_t*)audio->MessageQueueList.ptr)[audio->MessageQueueList.Current - 1];
			Message->Type = MessageType_t::ResumeGroup;
			Message->Data.ResumeGroup.GroupID = GroupID;
			TH_unlock(&audio->MessageQueueListMutex);
		}

		void audio_stop_group(audio_t* audio, uint32_t GroupID) {

		}

		void audio_set_volume(audio_t* audio, f32_t Volume) {
			if (ma_device_set_master_volume(&audio->device, Volume) != MA_SUCCESS) {
				fan::throw_error("audio_set_volume");
			}
		}
		f32_t audio_get_volume(audio_t* audio) {
			f32_t Volume;
			if (ma_device_get_master_volume(&audio->device, &Volume) != MA_SUCCESS) {
				fan::throw_error("audio_get_volume");
			}
			return Volume;
		}

		uint32_t SoundPlay(audio_t* audio, piece_t* piece, uint32_t GroupID, const PropertiesSoundPlay_t* Properties) {
#if fan_debug
			if (GroupID >= audio->GroupAmount) {
				fan::throw_error("fan_debug");
			}
#endif
			TH_lock(&audio->PlayInfoListMutex);
			PlayInfoList_NodeReference_t PlayInfoReference = PlayInfoList_NewNode(&audio->PlayInfoList);
			PlayInfoList_Node_t* PlayInfoNode = PlayInfoList_GetNodeByReference(&audio->PlayInfoList, PlayInfoReference);
			PlayInfoNode->data.piece = piece;
			PlayInfoNode->data.GroupID = GroupID;
			PlayInfoNode->data.PlayID = (uint32_t)-1;
			PlayInfoNode->data.properties = *Properties;
			PlayInfoNode->data.offset = 0;
			PlayInfoList_linkPrev(&audio->PlayInfoList, audio->GroupList[GroupID].LastReference, PlayInfoReference);
			TH_unlock(&audio->PlayInfoListMutex);

			TH_lock(&audio->MessageQueueListMutex);
			VEC_handle0(&audio->MessageQueueList, 1);
			Message_t* Message = &((Message_t*)audio->MessageQueueList.ptr)[audio->MessageQueueList.Current - 1];
			Message->Type = MessageType_t::SoundPlay;
			Message->Data.SoundPlay.PlayInfoReference = PlayInfoReference;
			TH_unlock(&audio->MessageQueueListMutex);

			return PlayInfoReference;
		}

		namespace {
			static holder_t get_file_info(const std::string& path) {

				holder_t holder;

				int error = 0;
				holder.decoder = op_open_file(path.c_str(), &error);
				if (error != 0) {
					opus_print_error(error);
					throw std::runtime_error("failed to open opus file");
				}

				holder.head = (OpusHead*)op_head(holder.decoder, 0);

				return holder;
			}
		}
		void piece_open_from_file(fan::audio::audio_t* audio, fan::audio::piece_t* piece, const std::string& path) {
			piece->audio = audio;
			piece->holder = get_file_info(path);
			piece->raw_size = op_pcm_total(piece->holder.decoder, 0);
			uint32_t CacheAmount = piece->raw_size / constants::FrameCacheAmount + !!(piece->raw_size % constants::FrameCacheAmount);
			piece->Cache = (PieceCache_t*)A_resize(0, CacheAmount * sizeof(PieceCache_t));
			for (uint32_t i = 0; i < CacheAmount; i++) {
				piece->Cache[i].ref = (FrameCacheList_NodeReference_t)-1;
			}
		}
	}
}
