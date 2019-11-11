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

// TODO: Actual ray bouncing and material support

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

#[derive(Debug, Default, Clone, Copy)]
struct Triangle {
    v0: Vec3,
    v1: Vec3,
    v2: Vec3,
    n0: Vec3,
    n1: Vec3,
    n2: Vec3,
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

    fn centroid(&self) -> Vec3 {
        Vec3 {
            x: (self.v0.x + self.v1.x + self.v2.x) / 3.0,
            y: (self.v0.y + self.v1.y + self.v2.y) / 3.0,
            z: (self.v0.z + self.v1.z + self.v2.z) / 3.0,
        }
    }
}

#[derive(Debug)]
struct AABB {
    min: Vec3,
    max: Vec3,
}

fn fmin(t0: f32, t1: f32) -> f32 {
    if t0 < t1 {
        t0
    } else {
        t1
    }
}

fn fmax(t0: f32, t1: f32) -> f32 {
    if t0 > t1 {
        t0
    } else {
        t1
    }
}

impl AABB {
    fn new() -> AABB {
        AABB {
            min: Vec3::zero(),
            max: Vec3::zero(),
        }
    }

    // Intersection test using 3D slab-method
    fn intersects(&self, ray: &Ray) -> bool {
        let min = (self.min - ray.origin) / ray.direction;
        let max = (self.max - ray.origin) / ray.direction;

        let mut tmin = 0.001;
        let mut tmax = std::f32::MAX; 

        {
            let t0 = fmin(min.x, max.x);
            let t1 = fmax(min.x, max.x);
            tmin = fmax(t0, tmin);
            tmax = fmin(t1, tmax);
            if tmax < tmin {
                return false;
            }
        }

        {
            let t0 = fmin(min.y, max.y);
            let t1 = fmax(min.y, max.y);
            tmin = fmax(t0, tmin);
            tmax = fmin(t1, tmax);
            if tmax < tmin {
                return false;
            }
        }

        {
            let t0 = fmin(min.z, max.z);
            let t1 = fmax(min.z, max.z);
            tmin = fmax(t0, tmin);
            tmax = fmin(t1, tmax);
            if tmax < tmin {
                return false;
            }
        }

        true
    }

    fn from_triangles(triangles: &[Triangle]) -> AABB {
        let mut min_x = std::f32::MAX;
        let mut min_y = std::f32::MAX;
        let mut min_z = std::f32::MAX;
        let mut max_x = std::f32::MIN;
        let mut max_y = std::f32::MIN;
        let mut max_z = std::f32::MIN;

        for triangle in triangles {
            // Min
            if triangle.v0.x < min_x {
                min_x = triangle.v0.x;
            }
            if triangle.v1.x < min_x {
                min_x = triangle.v1.x;
            }
            if triangle.v2.x < min_x {
                min_x = triangle.v2.x;
            }

            if triangle.v0.y < min_y {
                min_y = triangle.v0.y;
            }
            if triangle.v1.y < min_y {
                min_y = triangle.v1.y;
            }
            if triangle.v2.y < min_y {
                min_y = triangle.v2.y;
            }

            if triangle.v0.z < min_z {
                min_z = triangle.v0.z;
            }
            if triangle.v1.z < min_z {
                min_z = triangle.v1.z;
            }
            if triangle.v2.z < min_z {
                min_z = triangle.v2.z;
            }

            // Max
            if triangle.v0.x > max_x {
                max_x = triangle.v0.x;
            }
            if triangle.v1.x > max_x {
                max_x = triangle.v1.x;
            }
            if triangle.v2.x > max_x {
                max_x = triangle.v2.x;
            }

            if triangle.v0.y > max_y {
                max_y = triangle.v0.y;
            }
            if triangle.v1.y > max_y {
                max_y = triangle.v1.y;
            }
            if triangle.v2.y > max_y {
                max_y = triangle.v2.y;
            }

            if triangle.v0.z > max_z {
                max_z = triangle.v0.z;
            }
            if triangle.v1.z > max_z {
                max_z = triangle.v1.z;
            }
            if triangle.v2.z > max_z {
                max_z = triangle.v2.z;
            }
        }

        AABB {
            min: Vec3 {
                x: min_x,
                y: min_y,
                z: min_z,
            },
            max: Vec3 {
                x: max_x,
                y: max_y,
                z: max_z,
            },
        }
    }

    fn is_point_inside(&self, p: Vec3) -> bool {
        if p.x > self.max.x {
            return false;
        }
        if p.y > self.max.y {
            return false;
        }
        if p.z > self.max.z {
            return false;
        }
        if p.x < self.min.x {
            return false;
        }
        if p.y < self.min.y {
            return false;
        }
        if p.z < self.min.z {
            return false;
        }
        true
    }

    fn subdivide(&self, dimension: usize) -> Vec<AABB> {
        let size = (self.max - self.min) / (dimension as f32);
        let mut aabbs = Vec::new();

        for i in 0..dimension {
            for j in 0..dimension {
                for k in 0..dimension {
                    let offset = Vec3 {
                        x: i as f32,
                        y: j as f32,
                        z: k as f32,
                    };
                    let min = self.min + size * offset;
                    let aabb = AABB {
                        min: min,
                        max: min + size,
                    };
                    println!("{:?}", aabb);
                    aabbs.push(aabb);
                }
            }
        }
        aabbs
    }
}

#[derive(Debug)]
struct Octree {
    dimension: usize,
    aabbs: Vec<AABB>,
    triangles: Vec<Vec<Triangle>>,
}

impl Octree {
    fn new(dimension: usize, triangles: Vec<Triangle>) -> Octree {
        let root_aabb = AABB::from_triangles(&triangles);
        println!("Root AABB: {:?}", root_aabb);
        let aabbs = root_aabb.subdivide(dimension);

        // FIXME: Places triangles in correct AABB based on centroid location
        let mut aabb_triangles: Vec<Vec<Triangle>> = Vec::new();
        for _ in 0..dimension * dimension * dimension {
            aabb_triangles.push(Vec::new());
        }

        for triangle in triangles {
            let centroid = triangle.centroid();
            for (i, aabb) in aabbs.iter().enumerate() {
                if aabb.is_point_inside(centroid) {
                    aabb_triangles[i].push(triangle);
                    break;
                }
            }
        }

        Octree {
            dimension: dimension,
            aabbs: aabbs,
            triangles: aabb_triangles,
        }
    }

    fn intersects(&self, ray: &Ray) -> RaycastResult {
        for (i, aabb) in self.aabbs.iter().enumerate() {
            let hit = aabb.intersects(ray);
            if hit {
                for triangle in &self.triangles[i] {
                    let res = triangle.intersects(ray);
                    match res {
                        RaycastResult::Hit { t, u, v } => {
                            return res;
                        },
                        _ => continue,
                    };
                }
            }
        }
        RaycastResult::Miss
    }
}

struct Scene {
    octrees: Vec<Octree>,
}

impl Scene {
    fn new(path: &str) -> Scene {
        let start = std::time::Instant::now();
        println!("Starting to load scene ...");

        let mut octrees = Vec::new();

        let obj = tobj::load_obj(&Path::new(path));
        assert!(obj.is_ok());
        let (models, materials) = obj.unwrap();

        println!("# of models: {}", models.len());
        println!("$ of materials: {}", materials.len());

        for (i, m) in models.iter().enumerate() {
            let mesh = &m.mesh;
            println!("model[{}].name = \'{}\'", i, m.name);
            println!("model[{}].mesh.material_id = {:?}", i, mesh.material_id);

            println!("model[{}].vertices: {}", i, mesh.positions.len() / 3);
            assert!(mesh.positions.len() % 3 == 0);

            println!("model[{}].indices: {}", i, mesh.indices.len());
            assert!(mesh.indices.len() % 3 == 0);

            println!("model[{}].triangles: {}", i, mesh.indices.len() / 3);

            assert!(!mesh.normals.is_empty(), "Model lacks normals.");
            assert!(
                !mesh.texcoords.is_empty(),
                "Model lacks texture coordinates."
            );

            let mut triangles = Vec::new();
            for idxs in mesh.indices.chunks(3) {
                let mut v = [Vec3::default(); 3];
                for i in 0..3 {
                    let idx = idxs[i] as usize;
                    let x = mesh.positions[3 * idx];
                    let y = mesh.positions[3 * idx + 1];
                    let z = mesh.positions[3 * idx + 2];
                    v[i] = Vec3 { x: x, y: y, z: z }
                }

                let mut n = [Vec3::default(); 3];
                for i in 0..3 {
                    let idx = idxs[i] as usize;
                    let x = mesh.normals[3 * idx];
                    let y = mesh.normals[3 * idx + 1];
                    let z = mesh.normals[3 * idx + 2];
                    n[i] = Vec3 { x: x, y: y, z: z }
                }

                triangles.push(Triangle {
                    v0: v[0],
                    v1: v[1],
                    v2: v[2],
                    n0: n[0],
                    n1: n[1],
                    n2: n[2],
                });
            }

            let dimension = 4;
            octrees.push(Octree::new(dimension, triangles));
        }

        let end = std::time::Instant::now();
        let diff = end - start;
        println!(
            "Scene loaded in: {}.{} secs",
            diff.as_secs(),
            diff.subsec_millis()
        );

        Scene { octrees: octrees }
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
        for (i, octree) in self.octrees.iter().enumerate() {
            let res = octree.intersects(&ray);
            match res {
                RaycastResult::Hit { t, u, v } => {
                    return Vec3::new(0.5)
                },
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

    let scene = Scene::new("/home/alexander/Desktop/models/rust_logo.obj");
    let from = Vec3 {
        x: 0.0,
        y: 2.0,
        z: 2.0,
    };
    let to = Vec3::zero();
    let camera = Camera::new(from, to);

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let start = std::time::Instant::now();

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

        let end = std::time::Instant::now();
        let diff = end - start;
        println!(
            "Frame rendered in: {}.{} secs",
            diff.as_secs(),
            diff.subsec_millis()
        );

        window.update_with_buffer(&buffer).unwrap();
    }
}
