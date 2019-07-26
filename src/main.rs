extern crate minifb;

use minifb::{Key, WindowOptions, Window};

const WINDOW_WIDTH: usize = 640;
const WINDOW_HEIGHT: usize = 360;

struct Vec3 {
    pub x: f32, y: f32, z: f32
}

impl Vec3 {
    fn new(s: f32) -> Vec3 {
        Vec3{x: s, y: s, z: s}
    }

    fn zero() -> Vec3 {
        Vec3::new(0.0)
    }
}

struct Camera {
    position: Vec3,
    direction: Vec3
}

struct HitData {

}

enum RaycastResult {
    Miss,
    Hit(HitData)
}

struct Ray {
    pub origin: Vec3,
    pub direction: Vec3
}

impl Ray {
    fn new() -> Ray {
        // TODO: Impl.
        Ray{origin: Vec3::zero(), direction: Vec3::new(1.0)}
    }

    fn cast_ray() -> u32 {
        u32::max_value()
    }
} 

fn main() {
    let mut buffer: Vec<u32> = vec![0; WINDOW_WIDTH * WINDOW_HEIGHT];

    let mut window = Window::new("Test - ESC to exit",
                                 WINDOW_WIDTH,
                                 WINDOW_HEIGHT,
                                 WindowOptions::default()).unwrap_or_else(|e| {
                                     panic!("{}", e);
                                 });

    let camera = Camera::new();
    // let scene = Scene::load_json_from_file('~/Desktop/scene.json');

    while window.is_open() && !window.is_key_down(Key::Escape) {
        for i in buffer.iter_mut() {
            let ray = Ray::new(); // Direction from the camera
            *i = cast_ray();
        }

        // We unwrap here as we want this code to exit if it fails. Real applications may want to handle this in a different way
        window.update_with_buffer(&buffer).unwrap();
    }
}
