#define player_ ((OFFSETLESS(this, pile_t, peers))->player)

struct peers_t {
  peers_t() : client("127.0.0.1", 7777,
    // send callback
    [this]() { return *dynamic_cast<loco_t::shape_t*>(&player_.player); },
    // receive callback
    [this](const std::vector<loco_t::shape_t>& data) { peers = data; }
  ) 
  {

  }

  void step() {
    /*  */
  }

  fan::event::task_t global_task;
  fan::event::task_t write_task;
  fan::event::task_t listen_task;

  bool connected_to_server = false;
  fan::graphics::network::game_client_t<loco_t::shape_t> client;
  std::vector<loco_t::shape_t> peers;
};
#undef player_