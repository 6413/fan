#pragma once

#include <WITCH/WITCH.h>

#include <AL/al.h>
#include <AL/alc.h>

#include <fan/types/vector.hpp>
#include <fan/io/file.hpp>

#include <fan/time/time.hpp>

#include <thread>

#include <opus/opusfile.h>

#include <WITCH/EV/EV.h>
#include <WITCH/TH/TH.h>
#include <WITCH/VAS/VAS1.h>

#ifdef fan_platform_windows
	#include <endpointvolume.h>

	#include <mmdeviceapi.h>
	#include <endpointvolume.h>
#endif

#ifdef fan_compiler_visual_studio
	#pragma comment(lib, "lib/opus/opus.lib")
	#pragma comment(lib, "lib/ogg/libogg.lib")
	#pragma comment(lib, "lib/opus/opusfile.lib")
	#pragma comment(lib, "lib/WITCH/uv/uv.lib")
	#pragma comment(lib, "lib/al/OpenAL32.lib")
	#pragma comment(lib, "Winmm.lib")
#endif


namespace fan {
	namespace audio {
		namespace{
			namespace constants{
				static constexpr uint32_t mono = 1;
				static constexpr uint32_t stereo = 2;

				static constexpr f32_t default_volume = 1;

				// buffer size * channel count, make divideable with channel count
				inline static constexpr std::size_t buffer_size = 48000;
				inline static constexpr std::size_t number_of_buffers = 2;

				static constexpr uint32_t opus_decode_sample_rate = 48000;
			}
			static void al_check_error(void) {
				auto error = alGetError();
				if (error != AL_NO_ERROR) {
					fan::throw_error("openal error " + std::to_string(error));
				}
			}
			static uint32_t al_generate_source(void) {
				uint32_t source_reference;

				alGenSources(1, &source_reference);
				al_check_error();

				alSourcef(source_reference, AL_GAIN, constants::default_volume);
				al_check_error();

				alSourcef(source_reference, AL_MAX_GAIN, constants::default_volume);
				al_check_error();

				return source_reference;
			}
			static uint32_t al_get_format(uint32_t channel_count) {
				uint32_t format;

				switch (channel_count) {
					case constants::mono: {
						format = AL_FORMAT_MONO16;
						break;
					}
					case constants::stereo: {
						format = AL_FORMAT_STEREO16;
						break;
					}
					default: {
						throw std::runtime_error("channel layout invalid or not specified");
					}
				}

				return format;
			}

			static void opus_print_error(int error) {
				switch (error) {
				case OP_FALSE:  fan::print("OP_FALSE"); break;
				case OP_EOF:  fan::print("OP_EOF"); break;
				case OP_HOLE:  fan::print("OP_HOLE"); break;
				case OP_EREAD:  fan::print("OP_EREAD"); break;
				case OP_EFAULT:  fan::print("OP_EFAULT"); break;
				case OP_EIMPL:  fan::print("OP_EIMPL"); break;
				case AL_OUT_OF_MEMORY: fan::print("AL_OUT_OF_MEMORY"); break;
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

			static std::vector<short> decode(holder_t* holder, uint64_t offset) {

				std::vector<opus_int16> temp_buffer(constants::buffer_size);

				seek(holder, offset);

				std::vector<short> raw;

				int64_t to_read = constants::buffer_size * holder->head->channel_count;

				uint64_t sum = 0;

				while (to_read > 0) {
					int64_t read = op_read(holder->decoder, temp_buffer.data(), constants::buffer_size - sum, 0) * holder->head->channel_count;
					if (read <= 0) {
						break;
					}

					to_read -= read;
					sum += read;
					raw.insert(raw.end(), temp_buffer.data(), temp_buffer.data() + read);
				}

				if (raw.size() & 1) {
					raw.push_back(raw[raw.size() - 1]);
				}

				return raw;
			}
		}

		struct audio_t {
			ALCdevice *device;
			ALCcontext *context;

			EV_t listener;
			EV_timer_t timer;

			TH_mutex_t play_info_list_mutex;
			VAS1_t play_info_list;
		};

		struct piece_t {
			audio_t *audio;
			holder_t holder;
			uint64_t raw_size;
		};

		struct properties_play_t {
			bool loop = 0;
		};
		struct properties_stop_t{
			struct{
				uint32_t ThreadSafe : 1 = true;
			}Flags;
			struct FadeOut{
			private:
				uint32_t RawTime = 0;
			public:
				FadeOut operator=(f32_t Time){
					RawTime = Time * constants::opus_decode_sample_rate;
				}
				uint32_t GetRawTime(void){
					return RawTime;
				}
			}FadeOut;
		};

		namespace{
			struct play_info_t {
				uint32_t source_reference;
				piece_t *piece;

				properties_play_t properties;

				ALuint buffers[constants::number_of_buffers];
				uint64_t offset;
			};
			void internal_play_reference_stop(EV_t *listener, void *p){
				audio_t *audio = OFFSETLESS(listener, audio_t, listener);
				uint32_t play_reference = (uintptr_t)p;
				play_info_t *play_info = (play_info_t *)VAS1_out(&audio->play_info_list, play_reference);
				alSourceStop(play_info->source_reference);
				al_check_error();
				alDeleteSources(1, &play_info->source_reference);
				al_check_error();
				uint32_t TotalBuffers = play_info->piece->raw_size / constants::buffer_size;
				if(TotalBuffers > constants::number_of_buffers){
					TotalBuffers = constants::number_of_buffers;
				}
				else if(TotalBuffers == 0){
					TotalBuffers = 1;
				}
				alDeleteBuffers(TotalBuffers, play_info->buffers);
				al_check_error();
				TH_lock(&play_info->piece->audio->play_info_list_mutex);
				VAS1_unlink(&play_info->piece->audio->play_info_list, play_reference);
				TH_unlock(&play_info->piece->audio->play_info_list_mutex);
			}
		}
		void play_reference_stop(audio_t *audio, uint32_t play_reference, const properties_stop_t *properties_stop){
			if(properties_stop->Flags.ThreadSafe == true){
				EV_queue_lock(&audio->listener);
				EV_queue_add(&audio->listener, internal_play_reference_stop, (void *)play_reference);
				EV_queue_unlock(&audio->listener);
				EV_queue_signal(&audio->listener);
			}
			else{
				internal_play_reference_stop(&audio->listener, (void *)play_reference);
			}
		}
		namespace{
			static void buffer_checker_cb(EV_t *listener, EV_timer_t *t) {
				audio_t *audio = OFFSETLESS(listener, audio_t, listener);

				TH_lock(&audio->play_info_list_mutex);
				uint32_t LastReference = VAS1_GetNodeLast(&audio->play_info_list);
				while(LastReference != audio->play_info_list.src){
					play_info_t *play_info = (play_info_t *)VAS1_out(&audio->play_info_list, LastReference);
					if(play_info->source_reference != (ALuint)-1){
						LastReference = *VAS1_road0(&audio->play_info_list, LastReference);
						break;
					}
					LastReference = *VAS1_road1(&audio->play_info_list, LastReference);
				}
				if(LastReference == audio->play_info_list.src){
					LastReference = *VAS1_road0(&audio->play_info_list, LastReference);
				}
				TH_unlock(&audio->play_info_list_mutex);

				uint32_t reference = VAS1_GetNodeFirst(&audio->play_info_list);
				while(reference != LastReference){
					play_info_t *play_info = (play_info_t *)VAS1_out(&audio->play_info_list, reference);

					if(play_info->offset >= play_info->piece->raw_size){
						uint32_t NextReference = *VAS1_road0(&audio->play_info_list, reference);
						ALint queued = 0;
						alGetSourcei(play_info->source_reference, AL_BUFFERS_PROCESSED, &queued);
						al_check_error();
						if(queued == 0){
							properties_stop_t properties_stop;
							properties_stop.Flags.ThreadSafe = false;
							play_reference_stop(play_info->piece->audio, reference, &properties_stop);
						}
						else{
							ALint processed = 0;
							alGetSourcei(play_info->source_reference, AL_BUFFERS_PROCESSED, &processed);
							al_check_error();
							ALuint buffer[constants::number_of_buffers];
							alSourceUnqueueBuffers(play_info->source_reference, processed, buffer);
							al_check_error();
							if(processed == queued){
								properties_stop_t properties_stop;
								properties_stop.Flags.ThreadSafe = false;
								play_reference_stop(play_info->piece->audio, reference, &properties_stop);
							}
						}
						reference = NextReference;
						continue;
					}

					ALint processed = 0;
					alGetSourcei(play_info->source_reference, AL_BUFFERS_PROCESSED, &processed);
					al_check_error();

					while(processed--){
						ALuint buffer;
						alSourceUnqueueBuffers(play_info->source_reference, 1, &buffer);
						al_check_error();

						if(play_info->offset >= play_info->piece->raw_size){
							break;
						}
						auto raw = decode(&play_info->piece->holder, play_info->offset);

						alBufferData(buffer, al_get_format(play_info->piece->holder.head->channel_count), raw.data(), raw.size() * 2, constants::opus_decode_sample_rate);
						al_check_error();

						alSourceQueueBuffers(play_info->source_reference, 1, &buffer);
						al_check_error();

						play_info->offset += raw.size();
					}

					reference = *VAS1_road0(&audio->play_info_list, reference);
				}
			}
		}

		void audio_open(audio_t *audio) {
			EV_open(&audio->listener);

			TH_mutex_init(&audio->play_info_list_mutex);
			VAS1_open(&audio->play_info_list, sizeof(play_info_t), 0xfff);

			audio->device = alcOpenDevice(NULL);
			if (!audio->device) {
				fan::throw_error("no sound device " + std::to_string(alGetError()));
			}

			audio->context = alcCreateContext(audio->device, 0);
			if (!audio->context) {
				fan::throw_error("no sound context");
			}
			alcMakeContextCurrent(audio->context);

			EV_timer_init(&audio->timer, constants::buffer_size / constants::opus_decode_sample_rate, buffer_checker_cb);
			EV_timer_start(&audio->listener, &audio->timer);

			std::thread([address = (uintptr_t)&audio->listener]{
				EV_start((EV_t*)address);
			}).detach();
		}

		void audio_close(audio_t *audio) {
			VAS1_close(&audio->play_info_list);
			alcDestroyContext(audio->context);
			alcCloseDevice(audio->device);
		}

		uint32_t piece_play(audio_t *audio, piece_t *piece, const properties_play_t *properties) {
			TH_lock(&audio->play_info_list_mutex);
			uint32_t play_reference = VAS1_NewNodeLast(&audio->play_info_list);
			play_info_t *play_info = (play_info_t *)VAS1_out(&audio->play_info_list, play_reference);
			play_info->source_reference = (ALuint)-1;
			TH_unlock(&audio->play_info_list_mutex);

			play_info->piece = piece;
			play_info->offset = 0;
			play_info->properties = *properties;
			EV_queue_lock(&audio->listener);
			EV_queue_add(&audio->listener, [] (EV_t *l, void *t) {

				play_info_t *play_info = (play_info_t *)t;

				play_info->source_reference = al_generate_source();

				uint32_t TotalBuffers = play_info->piece->raw_size / constants::buffer_size;
				if(TotalBuffers > constants::number_of_buffers){
					TotalBuffers = constants::number_of_buffers;
				}
				else if(TotalBuffers == 0){
					if(play_info->piece->raw_size == 0){
						fan::throw_error("broken piece");
					}
					TotalBuffers = 1;
				}

				for (uint8_t i = 0; i < TotalBuffers; i++) {
					auto raw = decode(&play_info->piece->holder, play_info->offset);

					alGenBuffers(1, &play_info->buffers[i]);
					al_check_error();

					alBufferData(play_info->buffers[i], al_get_format(play_info->piece->holder.head->channel_count), raw.data(), raw.size() * 2, constants::opus_decode_sample_rate);
					al_check_error();

					play_info->offset += raw.size();
				}

				alSourceQueueBuffers(play_info->source_reference, TotalBuffers, play_info->buffers);
				al_check_error();

				alSourcePlay(play_info->source_reference);
				al_check_error();
			}, play_info);
			EV_queue_unlock(&audio->listener);
			EV_queue_signal(&audio->listener);

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
			if (!fan::io::file::exists(path)) {
				fan::throw_error("file does not exist \"" + path + "\"");
			}

			piece->holder = get_file_info(path);

			piece->raw_size = op_pcm_total(piece->holder.decoder, 0);
			piece->audio = audio;
		}

	}
}
