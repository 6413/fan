1. Non-player character automatically solving any maze (made could be randomly generated)

Visual demonstration of A* algorithm using a ready library
Made using own game engine in C++

Keybinds
Mouse Left - First click sets the starting position and second click sets the ending position
Mouse Right - adds or removes walls
Keyboard v - toggles vsync

Map
editing file "map" is allowed
value 0 representing a block without a wall and value 1 represents a wall
maximum size for map is 32x32

Known bugs
In the A* algorithm library I am using there isn't a possibility to check if the path is solveable,
in this case its undefined behaviour if you try to find a path for it

Two renderer backends same program
opengl_pathfind.exe runs in older gpus and in newer gpus
vulkan_pathfind.exe runs in newer gpus