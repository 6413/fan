typedef enum{
  IO_pipe_Flag_Packet = 0x01,
  IO_pipe_Flag_NonblockRead = 0x02,
  IO_pipe_Flag_NonblockWrite = 0x04
}IO_pipe_Flag;

enum AnimalFlags
{
    HasClaws   = 1,
    CanFly     = 2,
    EatsFish   = 4,
    Endangered = 8
};

  inline AnimalFlags operator|(IO_pipe_Flag p, IO_pipe_Flag b){
    return static_cast<AnimalFlags>(a | b);
  }

  int main() {

  }