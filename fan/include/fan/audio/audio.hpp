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

				static constexpr f32_t default_volume = 0.1;

				// buffer size * channel count, make divideable with channel count
				inline static constexpr std::size_t buffer_size = 81920;
				inline static constexpr std::size_t number_of_buffers = 4;

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

				alSourcei(source_reference, AL_SOURCE_RELATIVE, AL_TRUE);
				al_check_error();
				#if 0
				alGetListenerfv(AL_POSITION, listener_position.data());
				alGetListenerfv(AL_VELOCITY, listener_velocity.data());
				alGetListenerfv(AL_ORIENTATION, (float*)&listener_orientation[0][0]); // not sure
				alSourcef(source, AL_PITCH, 1.0);
				#endif
				alSourcef(source_reference, AL_GAIN, constants::default_volume);
				al_check_error();
				alSourcef(source_reference, AL_PITCH, 1);
				al_check_error();
				fan::vec3 p;
				alSourcefv(source_reference, AL_POSITION, p.data());
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
				if (int error = op_pcm_seek(holder->decoder, offset / (sizeof(opus_int16) * holder->head->channel_count))) {
					opus_print_error(error);
					throw std::runtime_error("failed to seek audio file");
				}
			}

			static std::vector<short> decode(holder_t* holder, uint32_t offset) {

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

			TH_mutex_t play_info_list_mutex;
			VAS1_t play_info_list;
		};

		struct piece_t {
			audio_t *audio;
			holder_t holder;
			uint64_t raw_size;
		};

		struct properties_play_t {
			bool loop;
			bool fadeout;
		};
		void properties_play_init(properties_play_t *p){
			p->loop = false;
		}

		namespace{
			struct play_info_t {
				EV_timer_t timer;
				uint32_t source_reference;
				piece_t *piece;

				properties_play_t properties;

				uint32_t sources;
				ALuint buffers[constants::number_of_buffers];
				uint64_t offset;
				uint64_t current_play_time; // nanoseconds
				uint64_t duration; // nanoseconds
			};
		}

		void audio_open(audio_t *audio) {
			EV_open(&audio->listener);

			TH_mutex_init(&audio->play_info_list_mutex);
			VAS1_open(&audio->play_info_list, sizeof(play_info_t), 0xfff);

			std::thread([address = (uintptr_t)&audio->listener]{
				EV_start((EV_t*)address);
			}).detach();

			audio->device = alcOpenDevice(NULL);
			if (!audio->device) {
				fan::throw_error("no sound device " + std::to_string(alGetError()));
			}

			// dont use sanitizer
			audio->context = alcCreateContext(audio->device, 0);
			if (!audio->context) {
				fan::throw_error("no sound context");
			}
			alcMakeContextCurrent(audio->context);
		}

		void audio_close(audio_t *audio) {
			VAS1_close(&audio->play_info_list);
			alcDestroyContext(audio->context);
			alcCloseDevice(audio->device);
		}

		void play_reference_stop(audio_t *audio, uint32_t play_reference){
			play_info_t *play_info = (play_info_t *)VAS1_out(&audio->play_info_list, play_reference);
			EV_timer_stop(&audio->listener, &play_info->timer);
			alSourceStop(play_info->source_reference);
			al_check_error();
			alDeleteSources(1, &play_info->source_reference);
			al_check_error();
			int n_buffers = 0;
			for (int i = 0; i < constants::number_of_buffers; i++) {
				if (play_info->buffers[i] == (ALuint)-1) {
					break;
				}
				n_buffers++;
			}
			alDeleteBuffers(n_buffers, play_info->buffers);
			al_check_error();
			TH_lock(&play_info->piece->audio->play_info_list_mutex);
			VAS1_unlink(&play_info->piece->audio->play_info_list, play_reference);
			TH_unlock(&play_info->piece->audio->play_info_list_mutex);
		}

		namespace{
			bool is_playing(uint32_t source) {
				int state;
				alGetSourcei(source, AL_SOURCE_STATE, &state);
				al_check_error();
				return state == AL_PLAYING;
			}
			static bool update_buffer(play_info_t *play_info) {
				if (!is_playing(play_info->source_reference)) {
					return false;
				}

				int processed = 0;

				while (processed <= 0) {
					alGetSourcei(play_info->source_reference, AL_BUFFERS_PROCESSED, &processed);
					al_check_error();
					if (processed == 0) {
						fan::print("we entered very bad place");
						fan::delay(fan::time::nanoseconds(10000));
					}
				}

				while (processed--) {
					ALuint buffer;
					alSourceUnqueueBuffers(play_info->source_reference, 1, &buffer);
					al_check_error();

					auto raw = decode(&play_info->piece->holder, play_info->offset);

					alBufferData(buffer, al_get_format(play_info->piece->holder.head->channel_count), raw.data(), raw.size(), constants::opus_decode_sample_rate);
					al_check_error();

					alSourceQueueBuffers(play_info->source_reference, 1, &buffer);
					al_check_error();

					play_info->offset += raw.size();
				}

				return true;
			}

			static void time_cb(EV_t *listener, EV_timer_t *t) {
				play_info_t *play_info = (play_info_t *)t;

				if (update_buffer(play_info)) {
					EV_timer_stop(listener, t);
					auto offset = constants::buffer_size + play_info->offset;

					EV_timer_init(t, (f32_t(play_info->piece->raw_size < offset ? play_info->piece->raw_size : offset) / 
						(constants::opus_decode_sample_rate * play_info->piece->holder.head->channel_count)), time_cb);
					EV_timer_start(listener, t);
				}
				else {
					if (play_info->properties.loop) {
						play_info->offset = 0;
						EV_timer_stop(listener, t);
						auto offset = constants::buffer_size + play_info->offset;
						EV_timer_init(t, (f32_t(constants::buffer_size < offset ? play_info->piece->raw_size : offset) / 
							(constants::opus_decode_sample_rate * play_info->piece->holder.head->channel_count)), time_cb);
						EV_timer_start(listener, t);

						int unque_buffers = 0;
						for (int i = 0; i < 4; i++) {
							if (play_info->buffers[i] == (ALint)-1) {
								break;
							}
							unque_buffers++;
						}

						alSourceUnqueueBuffers(play_info->source_reference, unque_buffers, play_info->buffers);

						int buffer_n = 0;
						for(std::size_t i = 0; i < constants::number_of_buffers; ++i)
						{
							auto raw = decode(&play_info->piece->holder, play_info->offset);
					
							alBufferData(play_info->buffers[i], al_get_format(play_info->piece->holder.head->channel_count), raw.data(), raw.size(), constants::opus_decode_sample_rate);
							al_check_error();

							play_info->offset += raw.size();

							buffer_n++;

							if (play_info->offset / 2 >= play_info->piece->raw_size) {
								break;
							}
						}

						alDeleteBuffers(4 - buffer_n, &play_info->buffers[buffer_n]);
						for(uint32_t i = buffer_n; i < constants::number_of_buffers; i++){
							play_info->buffers[i] = (ALuint)-1;
						}
						
						alSourceQueueBuffers(play_info->source_reference, buffer_n, play_info->buffers);
						al_check_error();

						alSourcePlay(play_info->source_reference);
						al_check_error();
						return;
					}

					uint32_t play_reference = ((uint8_t *)play_info - play_info->piece->audio->play_info_list.ptr) / (8 + sizeof(play_info_t));
					play_reference_stop(play_info->piece->audio, play_reference);
				}
			}
		}
		uint32_t piece_play(audio_t *audio, piece_t *piece, const properties_play_t *properties) {
			TH_lock(&audio->play_info_list_mutex);
			uint32_t play_reference = VAS1_NewNodeLast(&audio->play_info_list);
			TH_unlock(&audio->play_info_list_mutex);
			play_info_t *play_info = (play_info_t *)VAS1_out(&audio->play_info_list, play_reference);

			play_info->piece = piece;
			play_info->offset = 0;
			play_info->properties = *properties;
			play_info->current_play_time = fan::time::clock::now();
			play_info->duration = (f32_t)play_info->piece->raw_size / (constants::opus_decode_sample_rate * play_info->piece->holder.head->channel_count) * 1e+9;
			f32_t delay = f32_t(piece->raw_size < constants::buffer_size ? piece->raw_size : constants::buffer_size) / 
							(constants::opus_decode_sample_rate * play_info->piece->holder.head->channel_count);

			EV_timer_init(&play_info->timer, delay, time_cb);
			EV_queue_lock(&audio->listener);
			EV_queue_add(&audio->listener, [] (EV_t *l, void *t) {

				play_info_t* play_info = (play_info_t*)t;

				uint32_t source_reference = al_generate_source();

				play_info->source_reference = source_reference;

				alGenBuffers(constants::number_of_buffers, play_info->buffers);
				al_check_error();

				int buffer_n = 0;
				for(std::size_t i = 0; i < constants::number_of_buffers; ++i)
				{
					auto raw = decode(&play_info->piece->holder, play_info->offset);
				
					alBufferData(play_info->buffers[i], al_get_format(play_info->piece->holder.head->channel_count), raw.data(), raw.size(), constants::opus_decode_sample_rate);
					al_check_error();

					play_info->offset += raw.size();

					buffer_n++;

					if (play_info->offset / 2 >= play_info->piece->raw_size) {
						break;
					}
				}

				alDeleteBuffers(4 - buffer_n, &play_info->buffers[buffer_n]);
				for(uint32_t i = buffer_n; i < constants::number_of_buffers; i++){
					play_info->buffers[i] = (ALuint)-1;
				}

				alSourceQueueBuffers(play_info->source_reference, buffer_n, play_info->buffers);
				al_check_error();

				alSourcePlay(play_info->source_reference);
				al_check_error();

				EV_timer_start(l, &play_info->timer);
			}, play_info);
			EV_queue_unlock(&audio->listener);
			EV_queue_signal(&audio->listener);

			if (play_info->properties.fadeout) {
				EV_timer_init(&play_info->timer, 0.1, [](EV_t* listener, EV_timer_t* t) {
					play_info_t *play_info = (play_info_t *)t;

					// in seconds
					f32_t td = (f32_t)fan::time::clock::elapsed(play_info->current_play_time) / 1e+9;
					f32_t new_volume = constants::default_volume * (1.f - td);
					if (td >= play_info->duration / 1e+9 || new_volume <= 0) {
						EV_timer_stop(listener, t);
						return;
					}
					alSourcef(play_info->source_reference, AL_GAIN, new_volume);
				});
				EV_queue_lock(&audio->listener);
				EV_queue_add(&audio->listener, [](EV_t *l, void *t) {
					play_info_t *play_info = (play_info_t *)t;

					EV_timer_start(l, &play_info->timer);	
				}, play_info);
				EV_queue_unlock(&audio->listener);
				EV_queue_signal(&audio->listener);
			}

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

			piece->raw_size = op_pcm_total(piece->holder.decoder, 0) * piece->holder.head->channel_count;
			piece->audio = audio;
		}

	}
}
