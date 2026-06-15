#pragma once

namespace sf { class RenderWindow; }

// Switch an SFML window to native-resolution ("Retina") rendering and return the
// display's backing scale factor (2.0 on a typical Retina screen, 1.0 otherwise).
//
// SFML 2.6 never enables a high-resolution OpenGL surface on macOS (its Cocoa
// backend hard-codes highDpi = NO), so the window's framebuffer is allocated at
// point resolution and the OS upscales it on a Retina display -- which is why the
// graphics look fuzzy. This flips on the high-resolution surface so the framebuffer
// matches the physical pixels; the caller then scales the view's viewport by the
// returned factor so the existing (point-sized) coordinates fill the larger buffer.
//
// Returns 1.0 on non-Apple platforms (a no-op there).
float enableRetinaBacking(sf::RenderWindow& window);
