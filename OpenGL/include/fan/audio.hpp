#pragma once
#include <AL/al.h>
#include <AL/alc.h>

#include <FAN/types.h>
#include <FAN/file.hpp>

namespace fan {
	class audio {
	public:

		const static std::string temp_audio_file;
		const static std::string audio_info;
		const static std::string audio_input;
		const static const std::string audio_ffmpeg_format;

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

			fan::da_t<f32_t, 3> source_position(0, 0, 0);
			fan::da_t<f32_t, 3> source_velocity(0, 0, 0);
			fan::da_t<f32_t, 3> listener_position(0, 0, 0);
			fan::da_t<f32_t, 3> listener_velocity(-10, -10, 0);
			fan::da_t<f32_t, 3, 2> listener_orientation(0, 0, -1, 0, 1, 0);

			alSourcei(source, AL_SOURCE_RELATIVE, AL_TRUE);

			alGetListenerfv(AL_POSITION, listener_position.data());
			alGetListenerfv(AL_VELOCITY, listener_velocity.data());
			alGetListenerfv(AL_ORIENTATION, listener_orientation.data());

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

			file raw(temp_audio_file.c_str());
			raw.read();

			alSourcei(source, AL_BUFFER, 0);
			alBufferData(buffer, AL_FORMAT_MONO16, raw.data.data(), raw.data.size(), 44100);
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

		void set_position(const fan::da_t<f32_t, 3>& position) const {
			alSourcefv(source, AL_POSITION, position.data());
		}

		void set_velocity(const fan::da_t<f32_t, 3>& velocity) const {
			alSourcefv(source, AL_VELOCITY, velocity.data());
		}

		ALCdevice* device;
		ALCcontext* context;
		uint32_t source;
		uint32_t buffer;

	};

	const std::string audio::temp_audio_file = "sounds/temp.raw";
	const std::string audio::audio_info = " -f s16le -ar 44100 -ac 1 -acodec pcm_s16le ";
	const std::string audio::audio_input = " -i ";
	#ifdef FAN_WINDOWS
		const std::string audio::audio_ffmpeg_format = "\\ffmpeg";
	#else
		const std::string audio::audio_ffmpeg_format = "ffmpeg";
	#endif
}