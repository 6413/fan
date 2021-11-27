#pragma once

#include <AL/al.h>
#include <AL/alc.h>

#include <fan/types/vector.hpp>
#include <fan/io/file.hpp>

#include <fan/audio/audio_decoder.hpp>

#include <fan/time/time.hpp>

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

	}

	namespace audio {

		class audio_t {
		public:

			audio_t(const std::string& file) {

				device = alcOpenDevice(NULL);
				if (!device) {
					fan::print(alGetError());
					fan::print("no sound device");
					exit(1);
				}

				context = alcCreateContext(device, NULL);
				if (!context) {
					fan::print("no sound context");
					exit(1);
				}

				alcMakeContextCurrent(context);

				alGenSources(1, &source);
				alGenBuffers(1, &buffer);

				fan::audio_decoder::properties_t p;
				p.input = file;
				p.channel_layout = audio_decoder::channel_layout_e::mono;

				this->load_audio(fan::audio_decoder::decode(p));

				fan::vec3 source_position(0, 0, 0);
				fan::vec3 source_velocity(0, 0, 0);
				fan::vec3 listener_position(0, 0, 0);
				fan::vec3 listener_velocity(-10, -10, 0);
				fan::vec3 listener_orientation[2] = { fan::vec3(0, 0, -1), fan::vec3(0, 1, 0) };

				alSourcei(source, AL_SOURCE_RELATIVE, AL_TRUE);

/*				alGetListenerfv(AL_POSITION, listener_position.data());
				alGetListenerfv(AL_VELOCITY, listener_velocity.data());
				alGetListenerfv(AL_ORIENTATION, (float*)&listener_orientation[0][0]);*/ // not sure
																											alSourcef(source, AL_PITCH, 1.0);
																					  //alSourcef(source, AL_GAIN, 1.0);
				alSourcefv(source, AL_POSITION, source_position.data());
			}

			~audio_t() {
				alSourceStop(source);
				alSourcei(source, AL_BUFFER, 0);
				alDeleteBuffers(1, &buffer);
				alDeleteSources(1, &source);
				alcMakeContextCurrent(NULL);
				//alcDestroyContext(context);
				//alcCloseDevice(m_device);
			}

			bool is_playing() {
				int state;
				alGetSourcei(source, AL_SOURCE_STATE, &state);
				return state == AL_PLAYING;
			}

			void load_audio(audio_decoder::out_t out) {

				this->duration = out.duration;

				alSourcei(source, AL_BUFFER, 0);
				alBufferData(buffer, AL_FORMAT_MONO16, out.buffer.data(), out.buffer.size(), out.sample_rate);
				
				alSourcei(source, AL_BUFFER, buffer);
			}

			void play() {
				alSourcePlay(source);
				from_start_to_wait = fan::time::clock::now();
			}

			void pause() const {
				alSourcePause(source);
			}

			void stop() const {
				alSourceStop(source);
			}

			fan::vec3 get_position() const {
				fan::vec3 p;
				alGetListenerfv(AL_POSITION, p.data());

				return p;
			}
			void set_position(const fan::vec3& position) const {
				alSourcefv(source, AL_POSITION, position.data());
			}

			void set_velocity(const fan::vec3& velocity) const {
				alSourcefv(source, AL_VELOCITY, velocity.data());
			}

			// gets duration of current audio file in nanoseconds
			uint64_t get_duration() const {
				return duration;
			}

			// waits till audio is ended
			void wait() {
				fan::delay(fan::time::nanoseconds(duration - fan::time::clock::elapsed(from_start_to_wait)));
				from_start_to_wait = fan::time::clock::now();
			}

		protected:

			uint64_t from_start_to_wait = 0;

			uint64_t duration;

			ALCdevice* device;
			ALCcontext* context;

			uint32_t source;
			uint32_t buffer;

		};

	}
}