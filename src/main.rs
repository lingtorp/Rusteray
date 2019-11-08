extern crate minifb;

use std::path::Path;
extern crate tobj;

mod linalg;
use linalg::Vec3;

use minifb::{Key, Window, WindowOptions};

const WINDOW_WIDTH: usize = 400;
const WINDOW_HEIGHT: usize = 400;

struct Camera {
    frame_anchor: Vec3,
    u: Vec3,
    v: Vec3,
    position: Vec3,
}

impl Camera {
    fn new(from: Vec3, to: Vec3) -> Camera {
        let fov: f32 = 75.0;

        let aspect = (WINDOW_HEIGHT as f32) / (WINDOW_WIDTH as f32);

        let half_height = (fov.to_radians() / 2.0).tan();
        let half_width = aspect * half_height;

        let wup = Vec3::y();
        let dir = (from - to).normalize();

        let u = dir.cross(&wup).normalize();
        let v = dir.cross(&u).normalize();

        let horizontal = (2.0 * half_width) * u;
        let vertical = (2.0 * half_height) * v;

        let frame_anchor = from - half_width * u - half_height * v - dir;

        Camera {
            frame_anchor: frame_anchor,
            u: horizontal,
            v: vertical,
            position: from,
        }
    }

    // (s, t) screen based offsets
    fn ray(&self, s: f32, t: f32) -> Ray {
        let d = self.frame_anchor + s * self.u + t * self.v - self.position;
        Ray {
            origin: self.position,
            direction: d,
        }
    }
}

#[derive(Debug)]
enum RaycastResult {
    Miss,
    Hit { t: f32, u: f32, v: f32 },
}

#[derive(Debug)]
struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
}

#[derive(Debug)]
struct Triangle {
    v0: Vec3,
    v1: Vec3,
    v2: Vec3,
}

impl Triangle {
    // Ray-triangle intersection test from Real-Time Rendering 4th Edition (p. 964)
    fn intersects(&self, ray: &Ray) -> RaycastResult {
        let e1 = self.v1 - self.v0;
        let e2 = self.v2 - self.v0;
        let q = ray.direction.cross(&e2);

        let a = e1.dot(&q);
        if a < std::f32::EPSILON && a > std::f32::EPSILON {
            return RaycastResult::Miss;
        }

        let f = 1.0 / a;
        let s = ray.origin - self.v0;
        let u = f * (s.dot(&q));

        if u < 0.0 {
            return RaycastResult::Miss;
        }

        let r = s.cross(&e1);
        let v = f * (ray.direction.dot(&r));
        if v < 0.0 || u + v > 1.0 {
            return RaycastResult::Miss;
        }

        let t = f * (e2.dot(&r));
        RaycastResult::Hit { t, u, v }
    }
}

struct Scene {
    triangles: Vec<Triangle>,
}

impl Scene {
    fn new(path: &str) -> Scene {
        // let mut triangles: Vec<Triangle> = Vec::new();
        // triangles.push(Triangle {
        //     v0: Vec3 {
        //         x: 0.0,
        //         y: 0.5,
        //         z: 0.0,
        //     },
        //     v1: Vec3 {
        //         x: -0.5,
        //         y: 0.0,
        //         z: 0.0,
        //     },
        //     v2: Vec3 {
        //         x: 0.5,
        //         y: 0.0,
        //         z: 0.0,
        //     },
        // });
        // Scene {
        //     triangles: triangles,
        // }

        let obj = tobj::load_obj(&Path::new(path));
        assert!(obj.is_ok());
        let (models, materials) = obj.unwrap();

        println!("# of models: {}", models.len());
        println!("$ of materials: {}", materials.len());

        let mut triangles: Vec<Triangle> = Vec::new();

        for (i, m) in models.iter().enumerate() {
            let mesh = &m.mesh;
            println!("model[{}].name = \'{}\'", i, m.name);
            println!("model[{}].mesh.material_id = {:?}", i, mesh.material_id);

            println!("model[{}].vertices: {}", i, mesh.positions.len() / 3);
            assert!(mesh.positions.len() % 3 == 0);

            println!("model[{}].indices: {}", i, mesh.indices.len());
            assert!(mesh.indices.len() % 3 == 0);

            for idxs in mesh.indices.chunks(3) {
                let mut v = [Vec3::zero(), Vec3::zero(), Vec3::zero()];
                for i in 0..3 {
                    let idx = idxs[i] as usize;
                    let x = mesh.positions[3 * idx];
                    let y = mesh.positions[3 * idx + 1];
                    let z = mesh.positions[3 * idx + 2];
                    v[i] = Vec3 { x: x, y: y, z: z }
                }

                triangles.push(Triangle {
                    v0: v[0],
                    v1: v[1],
                    v2: v[2],
                });
            }
        }

        Scene {
            triangles: triangles,
        }
    }

    fn trace_background(&self, ray: &Ray) -> Vec3 {
        let t = ray.direction.y.abs();
        let white = Vec3::new(1.0);
        let light_blue = Vec3 {
            x: 0.5,
            y: 0.7,
            z: 1.0,
        };
        ((1.0 - t) * white + t * light_blue)
    }

    fn trace(&self, ray: &Ray) -> Vec3 {
        for triangle in &self.triangles {
            let res = triangle.intersects(&ray);
            match res {
                RaycastResult::Hit { t, u, v } => return Vec3::new(0.5),
                RaycastResult::Miss => continue,
            };
        }
        self.trace_background(ray)
    }
}

enum Encoding {
    ARGB,
}

fn encode_color(encoding: Encoding, color: Vec3) -> u32 {
    match encoding {
        Encoding::ARGB => {
            let ir = (color.x * 255.0) as u32;
            let ig = (color.y * 255.0) as u32;
            let ib = (color.z * 255.0) as u32;
            let ia = 1_u32;
            let mut pixel = 0_u32;
            pixel += ia << 24;
            pixel += ir << 16;
            pixel += ig << 8;
            pixel += ib;
            pixel
        }
    }
}

fn main() {
    let mut buffer: Vec<u32> = vec![0; WINDOW_WIDTH * WINDOW_HEIGHT];

    let mut window = Window::new(
        "Rusteray - ESC to exit",
        WINDOW_WIDTH,
        WINDOW_HEIGHT,
        WindowOptions::default(),
    )
    .unwrap_or_else(|e| {
        panic!("{}", e);
    });

    let scene = Scene::new("/home/alexander/Desktop/models/u_logo.obj");
    let from = Vec3 {
        x: 0.0,
        y: 0.0,
        z: 6.0,
    };
    let to = Vec3::zero();
    let camera = Camera::new(from, to);

    while window.is_open() && !window.is_key_down(Key::Escape) {
        for x in 0..WINDOW_WIDTH {
            for y in 0..WINDOW_HEIGHT {
                for s in 0..1 {
                    // Flipped view s.t y+ axis is up
                    let s = (y as f32) / (WINDOW_WIDTH as f32);
                    let t = (x as f32) / (WINDOW_HEIGHT as f32);
                    let ray = camera.ray(s, t);
                    let color = scene.trace(&ray);
                    buffer[x * WINDOW_WIDTH + y] = encode_color(Encoding::ARGB, color);
                }
            }
        }

        window.update_with_buffer(&buffer).unwrap();
    }
}
