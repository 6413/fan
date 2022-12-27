#include _WITCH_PATH(A/A.h)
#include _WITCH_PATH(MEM/MEM.h)
#include _WITCH_PATH(RAND/RAND.h)

typedef struct{
  uint16_t position_x;
  uint16_t position_y;
  uint16_t size_x;
  uint16_t size_y;
  uint32_t id;
}ETC_ImagePacker_Image_t;

typedef struct{
  uint16_t x;
  uint16_t y;
  uint32_t TotalImage;
  ETC_ImagePacker_Image_t *ImageList;
  uint8_t *ImagePixelData;
}ETC_ImagePacker_t;

void ETC_ImagePacker_new(
  ETC_ImagePacker_t *ImagePacker,
  uint16_t x,
  uint16_t y
){
  ImagePacker->x = x;
  ImagePacker->y = y;
  ImagePacker->TotalImage = 0;
  ImagePacker->ImageList = 0;
  ImagePacker->ImagePixelData = A_resize(0, x * y * 4);
}

uint32_t ETC_ImagePacker_get_index_by_id(
  ETC_ImagePacker_t *ImagePacker,
  uint32_t id
){
  for(uint32_t i = 0; i < ImagePacker->TotalImage; i++){
    ETC_ImagePacker_Image_t *image = &ImagePacker->ImageList[i];
    if(id == image->id){
      return i;
    }
  }
  return (uint32_t)-1;
}

bool _ETC_ImagePacker_is_position_size_collides(
  ETC_ImagePacker_t *ImagePacker,
  uint16_t position_x,
  uint16_t position_y,
  uint16_t size_x,
  uint16_t size_y
){
  uint8_t *image_data = A_resize(0, ImagePacker->x * ImagePacker->y);
  MEM_set(0, image_data, ImagePacker->x * ImagePacker->y);
  for(uint32_t i = 0; i < ImagePacker->TotalImage; i++){
    ETC_ImagePacker_Image_t *image = &ImagePacker->ImageList[i];
    for(uint16_t y = 0; y < image->size_y; y++){
      for(uint16_t x = 0; x < image->size_x; x++){
        image_data[(image->position_y + y) * ImagePacker->x + image->position_x + x] = 0xff;
      }
    }
  }
  for(uint16_t y = 0; y < size_y; y++){
    for(uint16_t x = 0; x < size_x; x++){
      if(image_data[(position_y + y) * ImagePacker->x + position_x + x]){
        A_resize(image_data, 0);
        return 1;
      }
    }
  }
  A_resize(image_data, 0);
  return 0;
}

void _ETC_ImagePacker_find_position_for_image(
  ETC_ImagePacker_t *ImagePacker,
  uint16_t size_x,
  uint16_t size_y,
  uint16_t *position_x,
  uint16_t *position_y
){
  uint16_t _position_x;
  uint16_t _position_y;
  do{
    _position_x = RAND_hard_32() % (ImagePacker->x - size_x);
    _position_y = RAND_hard_32() % (ImagePacker->y - size_y);
  }while(_ETC_ImagePacker_is_position_size_collides(ImagePacker, _position_x, _position_y, size_x, size_y));
  *position_x = _position_x;
  *position_y = _position_y;
}

void ETC_ImagePacker_add(
  ETC_ImagePacker_t *ImagePacker,
  uint16_t size_x,
  uint16_t size_y,
  uint32_t id,
  const void *ImagePixelData
){
  uint16_t position_x;
  uint16_t position_y;
  _ETC_ImagePacker_find_position_for_image(ImagePacker, size_x, size_y, &position_x, &position_y);
  for(uint16_t y = 0; y < size_y; y++){
    MEM_copy(((uint8_t *)ImagePixelData) + (y * size_x * 4), ImagePacker->ImagePixelData + (((position_y + y) * ImagePacker->x + position_x) * 4), size_x * 4);
  }
  ImagePacker->TotalImage++;
  ImagePacker->ImageList = (ETC_ImagePacker_Image_t *)A_resize(ImagePacker->ImageList, ImagePacker->TotalImage * sizeof(ETC_ImagePacker_Image_t));
  ETC_ImagePacker_Image_t *image = &ImagePacker->ImageList[ImagePacker->TotalImage - 1];
  image->position_x = position_x;
  image->position_y = position_y;
  image->size_x = size_x;
  image->size_y = size_y;
  image->id = id;
}

typedef struct{
  const uint8_t *ptr;
  uintptr_t size;
  uint8_t _step;
}ETC_ImagePacker_export_t;
void ETC_ImagePacker_export_init(
  ETC_ImagePacker_export_t *arg
){
  arg->_step = 0;
}
bool ETC_ImagePacker_export(
  ETC_ImagePacker_export_t *arg,
  const ETC_ImagePacker_t *ImagePacker
){
  step_switch:
  switch(arg->_step){
    case 0:{
      arg->ptr = (const uint8_t *)&ImagePacker->x;
      arg->size = sizeof(ImagePacker->x) + sizeof(ImagePacker->y) + sizeof(ImagePacker->TotalImage);
      break;
    }
    case 1:{
      arg->size = ImagePacker->TotalImage * sizeof(ETC_ImagePacker_Image_t);
      if(arg->size == 0){
        arg->_step++;
        goto step_switch;
      }
      arg->ptr = (const uint8_t *)ImagePacker->ImageList;
      break;
    }
    case 2:{
      arg->ptr = (const uint8_t *)ImagePacker->ImagePixelData;
      arg->size = ImagePacker->x * ImagePacker->y * 4;
      break;
    }
    case 3:{
      return 0;
    }
  }
  arg->_step++;
  return 1;
}
typedef struct{
  uint8_t *ptr;
  uintptr_t size;
  uint8_t _step;
}ETC_ImagePacker_import_t;
void ETC_ImagePacker_import_init(
  ETC_ImagePacker_import_t *arg
){
  arg->_step = 0;
}
bool ETC_ImagePacker_import(
  ETC_ImagePacker_import_t *arg,
  ETC_ImagePacker_t *ImagePacker
){
  step_switch:
  switch(arg->_step){
    case 0:{
      arg->ptr = (uint8_t *)&ImagePacker->x;
      arg->size = sizeof(ImagePacker->x) + sizeof(ImagePacker->y) + sizeof(ImagePacker->TotalImage);
      break;
    }
    case 1:{
      arg->size = ImagePacker->TotalImage * sizeof(ETC_ImagePacker_Image_t);
      if(arg->size == 0){
        ImagePacker->ImageList = 0;
        arg->_step++;
        goto step_switch;
      }
      ImagePacker->ImageList = (ETC_ImagePacker_Image_t *)A_resize(0, ImagePacker->TotalImage * sizeof(ETC_ImagePacker_Image_t));
      arg->ptr = (uint8_t *)ImagePacker->ImageList;
      break;
    }
    case 2:{
      ImagePacker->ImagePixelData = A_resize(0, ImagePacker->x * ImagePacker->y * 4);
      arg->ptr = (uint8_t *)ImagePacker->ImagePixelData;
      arg->size = ImagePacker->x * ImagePacker->y * 4;
      break;
    }
    case 3:{
      return 0;
    }
  }
  arg->_step++;
  return 1;
}
