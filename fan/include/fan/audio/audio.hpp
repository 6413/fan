#pragma once

#include <AL/al.h>
#include <AL/alc.h>

#include <fan/types/vector.hpp>
#include <fan/io/file.hpp>

#include <fan/time/time.hpp>

#include <thread>

#include <opus/opusfile.h>

#include <mutex>

#include <fan/timer_event.hpp>

#ifdef fan_compiler_visual_studio

	#pragma comment(lib, "lib/opus/opus.lib")
	#pragma comment(lib, "lib/ogg/libogg.lib")
	#pragma comment(lib, "lib/opus/opusfile.lib")

#endif

#ifdef fan_platform_windows

	#include <endpointvolume.h>

	#include <mmdeviceapi.h>
	#include <endpointvolume.h>

	#pragma comment(lib, "lib/al/OpenAL32.lib")
	#pragma comment(lib, "Winmm.lib")

#endif

namespace fan {

	namespace sys {

		class audio {
		public:

			audio() {

			#ifdef fan_platform_windows

				CoInitialize(0);
				IMMDeviceEnumerator *device_enumerator = 0;
				CoCreateInstance(__uuidof(MMDeviceEnumerator), 0, CLSCTX_INPROC_SERVER, __uuidof(IMMDeviceEnumerator), (LPVOID *)&device_enumerator);
				IMMDevice *default_device = 0;

				device_enumerator->GetDefaultAudioEndpoint(eRender, eConsole, &default_device);
				device_enumerator->Release();
				device_enumerator = 0;

				default_device->Activate(__uuidof(IAudioEndpointVolume), CLSCTX_INPROC_SERVER, 0, (LPVOID *)&end_point_volume);

				default_device->Release();
				default_device = 0; 

			#elif defined(fan_platform_unix)


			#endif

			}

			int get_master_volume() const {

				#ifdef fan_platform_windows

				f32_t current_volume = 0;
				end_point_volume->GetMasterVolumeLevelScalar(&current_volume);

				return (int)(current_volume * 100);

				#elif defined(fan_platform_unix)

				assert(0);

				#endif
			}

			void set_master_volume(int volume) {


				#ifdef fan_platform_windows

				end_point_volume->SetMasterVolumeLevelScalar((f32_t)volume / 100, 0);

				#elif defined(fan_platform_unix)

				assert(0);

				#endif

			}

		private:


			#ifdef fan_platform_windows

			IAudioEndpointVolume *end_point_volume = 0;

			#elif defined(fan_platform_unix)



			#endif

		};

	};

	namespace audio {

		static constexpr auto mono = 1;
		static constexpr auto stereo = 2;

		struct piece_base_t {

			piece_base_t() = default;

			using head_t = OpusHead*;

			struct holder_t {
				holder_t() = default;

				OggOpusFile* decoder;
				head_t head;
			};

			struct out_t {
				out_t() = default;

				std::vector<short> raw;

				head_t head;
			};

			static constexpr auto pitch = 1;

			struct properties_t {
				properties_t() = default;
				std::string input;
				uint32_t channel_layout = -1;
			};

			//~piece_base_t() {
			//	for (int i = 0; i < sources.size(); i++) {
			//		alSourceStop(sources[i]);
			//		alSourcei(sources[i], AL_BUFFER, 0);
			//	}
			//	alDeleteSources(sources.size(), sources.data());

			//	for (int i = 0; i < buffers.size(); i++) {
			//		alDeleteBuffers(buffers[i].size(), buffers[i].data());
			//	}

			//	alcMakeContextCurrent(NULL);
			//}

			bool is_playing(uint32_t source) {
				int state;
				alGetSourcei(source, AL_SOURCE_STATE, &state);
				return state == AL_PLAYING;
			}

		protected:

			static uint32_t get_al_format(auto channel_count) {
				uint32_t format;

				switch (channel_count) {
					case fan::audio::mono: {
						format = AL_FORMAT_MONO16;
						break;
					}
					case fan::audio::stereo: {
						format = AL_FORMAT_STEREO16;
						break;
					}
					default: {
						throw std::runtime_error("channel layout invalid or not specified");
					}
				}

				return format;
			}

			static void print_error(int error) {
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

		static holder_t get_file_info(const std::string& path) {

			holder_t holder;

			int error = 0;
			holder.decoder = op_open_file(path.c_str(), &error);

			if (error != 0) {
				print_error(error);
				throw std::runtime_error("failed to open opus file");
			}

			holder.head = (head_t)op_head(holder.decoder, 0);

			return holder;
		}

		static constexpr uint32_t ogg_sample_rate = 48000;

		static void seek(holder_t* holder, int64_t offset) {

			if (int error = op_pcm_seek(holder->decoder, offset)) {
				print_error(error);
				throw std::runtime_error("failed to seek audio file");
			}
		}

		public:

			uint32_t generate_source() {
				uint32_t source_reference;
				alGenSources(1, &source_reference);
				alSourcei(source_reference, AL_SOURCE_RELATIVE, AL_TRUE);

/*				alGetListenerfv(AL_POSITION, listener_position.data());
				alGetListenerfv(AL_VELOCITY, listener_velocity.data());
				alGetListenerfv(AL_ORIENTATION, (float*)&listener_orientation[0][0]);*/ // not sure
																											//alSourcef(source, AL_PITCH, 1.0);
				alSourcef(source_reference, AL_GAIN, 0.1);
				alSourcef(source_reference, AL_PITCH, pitch);
				fan::vec3 p;
				alSourcefv(source_reference, AL_POSITION, p.data());

				return source_reference;
			}

			void play(uint32_t source_reference) {
				alSourcePlay(source_reference);
				from_start_to_wait = fan::time::clock::now();
			}

			// buffer size * channel count, make divideable with channel count
			inline static constexpr std::size_t buffer_size = 81920;
			inline static constexpr std::size_t number_of_buffers = 4;

			bool update_stream(uint32_t index, uint32_t source_reference) {

				if (!is_playing(source_reference)) {
					return false;
				}

				if ((offset[index] + 1) * buffer_size >= raw_size) {
					return false;
				}

				int processed = 0;

				while (processed <= 0) {
					alGetSourcei(source_reference, AL_BUFFERS_PROCESSED, &processed);
					if (processed == 0) {
						fan::delay(fan::time::nanoseconds(10000));
					}
				}

				while (processed--) {
					ALuint buffer;
					alSourceUnqueueBuffers(source_reference, 1, &buffer);

					auto raw = this->decode(offset[index]);

					uint64_t size = 0;
					if ((offset[index] + 1) * buffer_size >= raw_size) {
						size = raw.size();
					}
					else {
						static constexpr auto pcm16 = 2;
						size = buffer_size * pcm16;
					}

					alBufferData(buffer, get_al_format(m_holder.head->channel_count), raw.data(), size, ogg_sample_rate);

					alSourceQueueBuffers(source_reference, 1, &buffer);

					offset[index]++;
				}

				return true;
			}

			std::vector<short> decode() {
				return this->decode(&m_holder);
			}

			std::vector<short> decode(uint32_t offset) {

				std::vector<opus_int16> temp_buffer(buffer_size);

				offset = buffer_size * offset / m_holder.head->channel_count;

				if (offset > raw_size / m_holder.head->channel_count - 1) {
					offset = raw_size / m_holder.head->channel_count - 1;
				}

				this->seek(&m_holder, offset);

				int64_t read;

				std::vector<short> raw;

				int64_t to_read = buffer_size * m_holder.head->channel_count;

				while (to_read > 0) {
					read = op_read(m_holder.decoder, temp_buffer.data(), buffer_size, 0);
					if (read <= 0) {
						break;
					}
					to_read -= read * m_holder.head->channel_count;
					raw.insert(raw.end(), temp_buffer.data(), temp_buffer.data() + read * m_holder.head->channel_count);
				}

				return raw;
			}

			std::vector<uint64_t> offset;

			std::mutex m_mutex;

			fan::event::timer_event_t te;

			void play_non_block_buffered(uint32_t* sr) {

				uint32_t source_reference = this->generate_source();

				*sr = source_reference;

				fan::print("+", source_reference);

				sources.emplace_back(source_reference);

				offset.emplace_back(number_of_buffers);

				buffers.resize(buffers.size() + 1);

				alGenBuffers(buffers[buffers.size() - 1].size(), buffers[buffers.size() - 1].data());

				uint64_t b_size;

				for(std::size_t i = 0; i < number_of_buffers; ++i)
				{
					auto raw = decode(i);
					
					static constexpr auto pcm16 = 2;

					b_size = buffer_size * pcm16;

					if (b_size > raw.size() * 2) {
						b_size = raw.size() * 2;
					}

					alBufferData(buffers[buffers.size() - 1][i], get_al_format(m_holder.head->channel_count), raw.data(), b_size, ogg_sample_rate);
				}

				alSourceQueueBuffers(source_reference, number_of_buffers, buffers[buffers.size() - 1].data());
				
				this->play(source_reference);

				auto amount = (f32_t(b_size) / 
							(ogg_sample_rate * m_holder.head->channel_count)) * 1e+9;
				te.push(amount, [&, current_offset = offset.size() - 1, source_reference] (uint64_t* time){

					if (update_stream(current_offset, source_reference)) {

						*time = (f32_t(buffer_size) / 
							(ogg_sample_rate * m_holder.head->channel_count)) * 1e+9;
							
					}
					else {
						this->stop(source_reference);
						*time = -1;
					}

				});

			}

			void pause(uint32_t source_reference) const {
				alSourcePause(source_reference);
			}

			void stop(uint32_t source_reference) {
				fan::print("-", source_reference);
				auto found = std::find(sources.begin(), sources.end(), source_reference);
				if (found == sources.end()) {
					fan::throw_error("invalid source " + source_reference);
				}

				auto d = std::distance(sources.begin(), found);
				assert(d < buffers.size() || d < offset.size());
				sources.erase(found);
				buffers.erase(buffers.begin() + d);
				offset.erase(offset.begin() + d);
				alSourceStop(source_reference);
				alDeleteSources(1, &source_reference);
			}

			fan::vec3 get_position() const {
				fan::vec3 p;
				alGetListenerfv(AL_POSITION, p.data());

				return p;
			}
			//void set_position(const fan::vec3& position) const {
			//	alSourcefv(source, AL_POSITION, position.data());
			//}

			//void set_velocity(const fan::vec3& velocity) const {
			//	alSourcefv(source, AL_VELOCITY, velocity.data());
			//}

			// gets duration of current audio file in nanoseconds
			uint64_t get_duration() const {
				return duration;
			}

			// waits till audio is ended
			void wait() {
				fan::delay(fan::time::nanoseconds(duration - fan::time::clock::elapsed(from_start_to_wait)));
				from_start_to_wait = fan::time::clock::now();
			}

			void wait(uint64_t custom_duration) {
				fan::delay(fan::time::nanoseconds(custom_duration - fan::time::clock::elapsed(from_start_to_wait)));
				from_start_to_wait = fan::time::clock::now();
			}

			static out_t decode(const properties_t& properties)
			{
				out_t out;

				int error = 0;
				auto opus_file = op_open_file(properties.input.c_str(), &error);

				out.head = (head_t)op_head(opus_file, 0);

				opus_int16* buffer = new opus_int16[0x2000];

				int read = 0;

				while ((read = op_read(opus_file, buffer, 0x2000, 0)) > 0) {
					out.raw.insert(out.raw.end(), buffer, buffer + read * out.head->channel_count);
				}

				delete[] buffer;

				return out;
			}

			void set_end_callback(std::function<void()> f) {
				m_end_callback = f;
			}

			void load(auto* audio, const std::string& path) {
				if (!fan::io::file::exists(path)) {
					fan::throw_error("file does not exist \"" + path + "\"");
				}

				te.threaded_update();

				m_holder = get_file_info(path);

				uint32_t format;

				raw_size = op_pcm_total(m_holder.decoder, 0) * m_holder.head->channel_count;
			}

	//	protected:

			static std::vector<short> decode(holder_t* holder)
			{
				std::vector<short> raw;

				opus_int16* buffer = new opus_int16[0x2000];

				int read = 0;

				seek(holder, 0);

				while ((read = op_read(holder->decoder, buffer, 0x2000, 0)) > 0) {
					raw.insert(raw.end(), buffer, buffer + read * holder->head->channel_count);
				}

				delete[] buffer;

				return raw;
			}

			uint64_t from_start_to_wait = 0;

			uint64_t duration;

			std::vector<uint32_t> sources;
			std::vector<std::array<uint32_t, number_of_buffers>> buffers;
			uint32_t current_buffer = 0;

			uint64_t raw_size = 0;

			std::function<void()> m_end_callback = []{};

			holder_t m_holder;

		};

		struct piece_t {

			void load(auto* audio, const std::string& path) {
				pbptr = new piece_base_t();

				pbptr->load(audio, path);
			}

			void free() {
				delete pbptr;
				pbptr = nullptr;
			}

			piece_base_t* pbptr;
		};

		struct audio_t {

			ALCdevice* device = nullptr;
			ALCcontext* context = nullptr;

			void init() {
				if (device == nullptr) {
					device = alcOpenDevice(NULL);
					if (!device) {
						fan::throw_error("no sound device " + std::to_string(alGetError()));
					}
				}

				if (context == nullptr) {
					// dont use sanitizer
					context = alcCreateContext(device, 0);
					if (!context) {
						fan::throw_error("no sound context");
					}
					alcMakeContextCurrent(context);
				}
			}

			void free() {
				alcDestroyContext(context);
				alcCloseDevice(device);
			}

			uint32_t play(piece_t* piece) {
				uint32_t source_reference;
				piece->pbptr->play_non_block_buffered(&source_reference);
				return source_reference;
			}

			void pause(uint32_t source_reference) {
				alSourcePause(source_reference);
			}

			void stop(piece_t* piece, uint32_t source_reference) {
				alSourceStop(source_reference);
				auto found = std::find(piece->pbptr->sources.begin(), piece->pbptr->sources.end(), source_reference);
				auto index = std::distance(piece->pbptr->sources.begin(), found);
				piece->pbptr->sources.erase(piece->pbptr->sources.begin() + index);
				piece->pbptr->buffers.erase(piece->pbptr->buffers.begin() + index);
				piece->pbptr->offset.erase(piece->pbptr->offset.begin() + index);
			}

		};

	}
}