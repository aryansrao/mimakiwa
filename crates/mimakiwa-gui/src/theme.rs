use iced::Color;

// Obsidian Chrome palette
pub const BG_MAIN: Color      = Color { r: 0.910, g: 0.906, b: 0.898, a: 1.0 }; // #E8E7E5 Alabaster
pub const BG_SIDEBAR: Color   = Color { r: 0.863, g: 0.859, b: 0.851, a: 1.0 }; // #DCDBD9
pub const BG_AI: Color        = Color { r: 0.882, g: 0.878, b: 0.871, a: 1.0 }; // #E1E0DE
pub const BG_USER: Color      = Color { r: 0.325, g: 0.408, b: 0.471, a: 1.0 }; // #536878 Blue Slate
pub const BG_INPUT: Color     = Color { r: 0.929, g: 0.925, b: 0.918, a: 1.0 }; // #EDECEB
pub const BG_HOVER: Color     = Color { r: 0.831, g: 0.827, b: 0.820, a: 1.0 }; // #D4D3D1

// Text
pub const TEXT: Color         = Color { r: 0.039, g: 0.039, b: 0.039, a: 1.0 }; // #0A0A0A Onyx
pub const TEXT_MUTED: Color   = Color { r: 0.420, g: 0.471, b: 0.502, a: 1.0 }; // #6B7880
pub const TEXT_LIGHT: Color   = Color { r: 0.961, g: 0.957, b: 0.949, a: 1.0 }; // #F5F4F2

// Borders
pub const BORDER: Color       = Color { r: 0.769, g: 0.765, b: 0.757, a: 1.0 }; // #C4C3C1

// Accent — Blue Slate
pub const ACCENT: Color       = Color { r: 0.325, g: 0.408, b: 0.471, a: 1.0 }; // #536878
pub const ACCENT_HOVER: Color = Color { r: 0.263, g: 0.337, b: 0.392, a: 1.0 }; // #435664

// Status
pub const GREEN: Color        = Color { r: 0.133, g: 0.769, b: 0.471, a: 1.0 };
pub const ORANGE: Color       = Color { r: 0.851, g: 0.529, b: 0.098, a: 1.0 };

pub fn confidence_color(conf: f32) -> Color {
    if conf >= 0.70 { GREEN }
    else if conf >= 0.40 { ORANGE }
    else { Color { r: 0.85, g: 0.22, b: 0.22, a: 1.0 } }
}

