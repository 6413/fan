void open(void* sod) {
    
}

void close() {

}

void update() {
  if (pile.loco.lighting.ambient < 1) {
    pile.loco.lighting.ambient += pile.loco.delta_time * 5;
  }
  else {
    pile.loco.lighting.ambient = 1;
  }
  pile.step();
}