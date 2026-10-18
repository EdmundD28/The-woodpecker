#pragma once
// Host stand-in for Adafruit GFX: font structs so the real font headers compile.
#include <cstdint>
#define PROGMEM
typedef struct { uint16_t bitmapOffset; uint8_t width, height, xAdvance; int8_t xOffset, yOffset; } GFXglyph;
typedef struct { uint8_t* bitmap; GFXglyph* glyph; uint16_t first, last; uint8_t yAdvance; } GFXfont;
