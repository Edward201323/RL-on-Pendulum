#include "render/retina.hpp"

#include <SFML/Graphics/RenderWindow.hpp>

#ifdef __APPLE__
#import <Cocoa/Cocoa.h>

float enableRetinaBacking(sf::RenderWindow& window) {
    // SFML's window handle is the NSWindow on macOS (SFWindowController returns it).
    NSWindow* nsWindow = static_cast<NSWindow*>(window.getSystemHandle());
    if (nsWindow == nil) return 1.f;

    NSView* view = [nsWindow contentView];
    if (view == nil) return 1.f;

    // Ask AppKit for a pixel-resolution backing store instead of the default
    // point-resolution one. (Apple insists on a literal YES here, not just non-zero.)
    [view setWantsBestResolutionOpenGLSurface:YES];

    // Nudge the active GL context so it re-reads the (now larger) drawable size --
    // otherwise the framebuffer stays at point resolution until the next resize.
    window.setActive(true);
    NSOpenGLContext* ctx = [NSOpenGLContext currentContext];
    if (ctx != nil) [ctx update];

    NSScreen* screen = [nsWindow screen];
    if (screen == nil) screen = [NSScreen mainScreen];
    return screen != nil ? static_cast<float>([screen backingScaleFactor]) : 1.f;
}

#else  // non-Apple: nothing to do.

float enableRetinaBacking(sf::RenderWindow&) { return 1.f; }

#endif
