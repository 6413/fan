#pragma once

#include <AL/al.h>
#include <AL/alc.h>

#include <fan/types/vector.hpp>
#include <fan/io/file.hpp>

#ifdef FAN_PLATFORM_WINDOWS

	#include <Mmsystem.h>

	#include <endpointvolume.h>

	#include <mmdeviceapi.h>
	#include <endpointvolume.h>

	#pragma comment(lib, "OpenAL32.lib")
	#pragma comment(lib, "Winmm.lib")

#endif

namespace fan {

	namespace sys {

		class audio {
		public:

			audio() {

			#ifdef FAN_PLATFORM_WINDOWS

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

			#elif defined(FAN_PLATFORM_UNIX)


			#endif

			}

			int get_master_volume() const {

				#ifdef FAN_PLATFORM_WINDOWS

				f32_t current_volume = 0;
				end_point_volume->GetMasterVolumeLevelScalar(&current_volume);

				return (int)(current_volume * 100);

				#elif defined(FAN_PLATFORM_UNIX)

				assert(0);

				#endif
			}

			void set_master_volume(int volume) {


				#ifdef FAN_PLATFORM_WINDOWS

				end_point_volume->SetMasterVolumeLevelScalar((f32_t)volume / 100, 0);

				#elif defined(FAN_PLATFORM_UNIX)

				assert(0);

				#endif

			}

		private:


			#ifdef FAN_PLATFORM_WINDOWS

			IAudioEndpointVolume *end_point_volume = 0;

			#elif defined(FAN_PLATFORM_UNIX)



			#endif

		};

	}

	namespace audio {

		class audio {
		public:

			const static std::string temp_audio_file;
			const static std::string audio_info;
			const static std::string audio_input;
			const static std::string audio_ffmpeg_format;

			audio(const std::string& file) : device(alcOpenDevice(NULL)), context(alcCreateContext(device, NULL)) {
				if (!device) {
					fan::print("no sound device");
					exit(1);
				}
				if (!context) {
					fan::print("no sound context");
					exit(1);
				}
				alcMakeContextCurrent(context);

				alGenSources(1, &source);
				alGenBuffers(1, &buffer);

				audio::load_audio(file);

				fan::vec3 source_position(0, 0, 0);
				fan::vec3 source_velocity(0, 0, 0);
				fan::vec3 listener_position(0, 0, 0);
				fan::vec3 listener_velocity(-10, -10, 0);
				fan::vec3 listener_orientation[2] = { fan::vec3(0, 0, -1), fan::vec3(0, 1, 0) };

				alSourcei(source, AL_SOURCE_RELATIVE, AL_TRUE);

				alGetListenerfv(AL_POSITION, listener_position.data());
				alGetListenerfv(AL_VELOCITY, listener_velocity.data());
				alGetListenerfv(AL_ORIENTATION, (float*)&listener_orientation[0][0]); // not sure

																					  //alSourcef(source, AL_PITCH, 1.0);
																					  //alSourcef(source, AL_GAIN, 1.0);
				alSourcefv(source, AL_POSITION, source_position.data());
			}

			~audio() {
				alSourceStop(source);
				alSourcei(source, AL_BUFFER, 0);
				alDeleteBuffers(1, &buffer);
				alDeleteSources(1, &source);
				alcMakeContextCurrent(NULL);
				//alcDestroyContext(context);
				//alcCloseDevice(device);
			}

			bool is_playing() {
				int state;
				alGetSourcei(source, AL_SOURCE_STATE, &state);
				return state == AL_PLAYING;
			}

			void load_audio(std::string file) {
				file.insert(0, "\"");
				file.insert(file.size(), "\"");
				std::string env = std::getenv("PATH");
				auto found = env.find("ffmpeg");
				#ifdef FAN_WINDOWS
					auto first = env.rfind(';', found);
					auto last = env.find(';', found);
					std::string str(env.begin() + first + 1, env.begin() + last);
					system((str + audio_ffmpeg_format + audio_input + file + audio_info + temp_audio_file).c_str());
				#else
					system((audio_ffmpeg_format + audio_input + file + audio_info + temp_audio_file).c_str());
				#endif

				std::string raw = fan::io::file::read(temp_audio_file);

				alSourcei(source, AL_BUFFER, 0);
				alBufferData(buffer, AL_FORMAT_MONO16, raw.data(), raw.size(), 44100);
				remove(temp_audio_file.c_str());
				alSourcei(source, AL_BUFFER, buffer);
			}

			void play() const {
				alSourcePlay(source);
			}

			void pause() const {
				alSourcePause(source);
			}

			void stop() const {
				alSourceStop(source);
			}

			void set_position(const fan::vec3& position) const {
				alSourcefv(source, AL_POSITION, position.data());
			}

			void set_velocity(const fan::vec3& velocity) const {
				alSourcefv(source, AL_VELOCITY, velocity.data());
			}

			ALCdevice* device;
			ALCcontext* context;
			uint32_t source;
			uint32_t buffer;

		};

	}

	const std::string audio::audio::temp_audio_file = "sounds/temp.raw";
	const std::string audio::audio::audio_info = " -f s16le -ar 44100 -ac 1 -acodec pcm_s16le ";
	const std::string audio::audio::audio_input = " -i ";
	#ifdef FAN_WINDOWS
		const std::string audio::audio_ffmpeg_format = "\\ffmpeg";
	#else
		const std::string audio::audio::audio_ffmpeg_format = "ffmpeg";
	#endif
}