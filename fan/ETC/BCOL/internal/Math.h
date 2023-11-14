static auto min(auto p0, auto p1){
  return p0 < p1 ? p0 : p1;
}
static auto max(auto p0, auto p1){
  return p0 > p1 ? p0 : p1;
}
static auto abs(auto p0){
  return p0 < 0 ? -p0 : p0;
}

template <uint8_t ts>
struct iterate_grid_for_circle_t{
  public:
    fan::vec_wrap_t<ts, sint32_t> gs; /* grid start */
  private:
    uint8_t c = 0;
    fan::vec_wrap_t<ts, sint32_t> ge[ts]; /* grid end */
    bool NeedInit = true;

    void InitCurrent(const auto &gbs, const auto &wp, f32_t er){
      if(c < ts){
        gs[c] = (wp[c] - er) / gbs[c];
        ge[c] = (wp[c] + er) / gbs[c];
      }
      if(c + 1 == ts){
        gs[c]--; /* we increase this even before check ge[c] so this is needed */
      }
    }
    void Increase(const auto &gbs, const auto &wp, f32_t er /* end radius */){
      c++;
      InitCurrent(gbs, wp, er);
    }
    void Decrease(){
      c--;
      if(c < ts){
        gs[c]++;
        if(gs[c] > ge[c]){
          Decrease();
        }
      }
    }

    static f32_t gbod(f32_t r, f32_t i0, f32_t i1){ /* get biggest of dimension */
      if(i0 <= 0 && i1 >= 0){
        return r;
      }
      f32_t d1 = min(min(abs(i0), abs(i1)) / r, (f32_t)1);
      f32_t d0 = std::sqrt((f32_t)1 - d1 * d1) * r;
      return d0;
    }
    bool _it(
      const auto &gbs, /* grid block size */
      const auto &wp, /* world position */
      f32_t r /* radius */
    ){
      while(1){
        if(c + 1 < ts){
          f32_t rp = (f32_t)gs[c] * gbs[c] - wp[c]; /* relative position */
          f32_t roff = gbod(r, rp, rp + gbs[c]); /* relative offset */
          Increase(gbs, wp, roff);
        }
        else if(c < ts){
          gs[c]++;
          if(gs[c] <= ge[c]){
            return true;
          }
          Decrease();
        }
        else{
          return false;
        }
      }
    }
  public:
    bool it(const auto &gbs, const auto &wp, f32_t r){
      if(NeedInit){
        NeedInit = false;
        InitCurrent(gbs, wp, r);
      }
      return _it(gbs, wp, r);
    }
};
