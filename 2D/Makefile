all: main

main: main.o input.o settings.o texture.o color.o
	g++ -g Source.cpp FAN/Input.cpp SOIL2/*.c FAN/Settings.cpp FAN/Texture.cpp FAN/Color.cpp -I../ -lglfw -lGL -lGLEW -I. -w -fpermissive

main.o: Source.cpp
	g++ -c Source.cpp -lglfw -lGL -lGLEW -I. -w

input.o: FAN/Input.cpp
	g++ -c FAN/Input.cpp -lglfw -lGL -lGLEW -w

settings.o: FAN/Settings.cpp
	g++ -c FAN/Settings.cpp -w

texture.o: FAN/Texture.cpp
	g++ -c FAN/Texture.cpp -I../ -lglfw -lGL -lGLEW -w
	
color.o: FAN/Color.cpp
	g++ -c FAN/Color.cpp -w