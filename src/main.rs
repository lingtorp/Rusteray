extern crate minifb;
use minifb::{Key, Window, WindowOptions};

extern crate tobj;

use std::path::Path;

extern crate rand;
use rand::prelude::*;

extern crate indicatif;

mod linalg;
use linalg::Vec3;

mod brdf;

use std::sync::Arc;

extern crate threadpool;
use threadpool::ThreadPool;

extern crate num_cpus;

use bluenoise::BlueNoise;
use rand_pcg::Pcg64Mcg;

use clap::Parser;

/// Simple program to greet a person
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Number of frames to render
    #[arg(short, long, default_value_t = 1)]
    frames: u8,
}

const WINDOW_WIDTH: usize = 600;
const WINDOW_HEIGHT: usize = 600;
const SAMPLES_PER_PIXEL: usize = 25;
const RAY_DEPTH_MAX: usize = 10;

#[derive(Debug, Copy, Clone)]
struct Camera {
    frame_anchor: Vec3,
    u: Vec3,
    v: Vec3,
    position: Vec3,
    fov: f32,
}

impl Camera {
    fn new(fov_degrees: f32, from: Vec3, to: Vec3) -> Camera {
        let aspect = (WINDOW_WIDTH as f32) / (WINDOW_HEIGHT as f32);

        let half_height = (fov_degrees.to_radians() / 2.0).tan();
        let half_width = aspect * half_height;

        let wup = Vec3::y();

        let w = (from - to).normalize();
        let u = w.cross(wup).normalize();
        let v = w.cross(u).normalize();

        let horizontal = 2.0 * half_width * u;
        let vertical = 2.0 * half_height * v;

        let frame_anchor = from - half_width * u - half_height * v - w;

        Camera {
            frame_anchor: frame_anchor,
            u: horizontal,
            v: vertical,
            position: from,
            fov: fov_degrees,
        }
    }

    // (s, t) screen based offsets
    fn ray(&self, s: f32, t: f32) -> Ray {
        let d = (self.frame_anchor + s * self.u + t * self.v - self.position).normalize();
        Ray {
            origin: self.position,
            direction: d,
            depth: 1,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Intersection {
    t: f32,          // ray.at(t) for intersection
    normal: Vec3,    // Interpolated normal vector at intersection
    tangent: Vec3,   // Interpolated tangent vector at intersection
    bitangent: Vec3, // Interpolated tangent vector at intersection
}

impl Intersection {
    // Transforms to intersection shading space from world space
    // Shading space = | (x: tangent, y: normal, z: bitangent) |
    fn to_shading(&self, v: Vec3) -> Vec3 {
        Vec3::from(
            self.tangent.dot(v),
            self.normal.dot(v),
            self.bitangent.dot(v),
        )
        .normalize()
    }
}

#[derive(Debug)]
enum RaycastResult {
    Miss,
    Hit(Intersection),
}

#[derive(Debug)]
struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
    pub depth: usize,
}

impl Ray {
    fn at(&self, t: f32) -> Vec3 {
        self.origin + t * self.direction
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct Triangle {
    v0: Vec3,
    v1: Vec3,
    v2: Vec3,
    n0: Vec3,
    n1: Vec3,
    n2: Vec3,
    t0: Vec3,
    t1: Vec3,
    t2: Vec3,
    b0: Vec3,
    b1: Vec3,
    b2: Vec3,
}

impl Triangle {
    // Ray-triangle intersection test from Real-Time Rendering 4th Edition (p. 964) (Möller-Trumbore algorithm)
    // Non-backface culling version of the algorithm
    // Src [0]: http://www.graphics.cornell.edu/pubs/1997/MT97.pdf
    fn intersects(&self, ray: &Ray) -> RaycastResult {
        let e1 = self.v1 - self.v0;
        let e2 = self.v2 - self.v0;
        let q = ray.direction.cross(e2);

        let a = e1.dot(q); // Determinant
        if a < std::f32::EPSILON && a > -std::f32::EPSILON {
            return RaycastResult::Miss;
        }

        let f = 1.0 / a;
        let s = ray.origin - self.v0;
        let u = f * (s.dot(q));

        if u < 0.0 || u > 1.0 {
            return RaycastResult::Miss;
        }

        let r = s.cross(e1);
        let v = f * (ray.direction.dot(r));
        if v < 0.0 || u + v > 1.0 {
            return RaycastResult::Miss;
        }

        let t = f * (e2.dot(r));

        // Line intersection - not Ray
        if t < 0.0 {
            return RaycastResult::Miss;
        }
        let intersection = Intersection {
            t: t,
            normal: self.normal_at(u, v),
            tangent: self.tangent_at(u, v),
            bitangent: self.bitangent_at(u, v),
        };

        RaycastResult::Hit(intersection)
    }

    // Returns normal interpolated across the triangle
    fn normal_at(&self, u: f32, v: f32) -> Vec3 {
        ((1.0 - u - v) * self.n0 + u * self.n1 + v * self.n2).normalize()
    }

    // Returns the tangent interpolated across the triangle
    fn tangent_at(&self, u: f32, v: f32) -> Vec3 {
        ((1.0 - u - v) * self.t0 + u * self.t1 + v * self.t2).normalize()
    }

    // Returns the bitangent interpolated across the triangle
    fn bitangent_at(&self, u: f32, v: f32) -> Vec3 {
        ((1.0 - u - v) * self.b0 + u * self.b1 + v * self.b2).normalize()
    }
}

#[derive(Debug)]
struct AABB {
    min: Vec3,
    max: Vec3,
}

/// Describes the degree of containment of geometric bodies
#[derive(Debug)]
enum Containment {
    None,
    Partial,
    Full,
}

impl AABB {
    // AABB-Ray intersection test using 3D slab-method
    fn intersects(&self, ray: &Ray) -> RaycastResult {
        let min = (self.min - ray.origin) / ray.direction;
        let max = (self.max - ray.origin) / ray.direction;

        let mut tmin = std::f32::MIN;
        let mut tmax = std::f32::MAX;

        {
            let t0 = linalg::fmin(min.x, max.x);
            let t1 = linalg::fmax(min.x, max.x);
            tmin = linalg::fmax(t0, tmin);
            tmax = linalg::fmin(t1, tmax);
            if tmax < tmin {
                return RaycastResult::Miss;
            }
        }

        {
            let t0 = linalg::fmin(min.y, max.y);
            let t1 = linalg::fmax(min.y, max.y);
            tmin = linalg::fmax(t0, tmin);
            tmax = linalg::fmin(t1, tmax);
            if tmax < tmin {
                return RaycastResult::Miss;
            }
        }

        {
            let t0 = linalg::fmin(min.z, max.z);
            let t1 = linalg::fmax(min.z, max.z);
            tmin = linalg::fmax(t0, tmin);
            tmax = linalg::fmin(t1, tmax);
            if tmax < tmin {
                return RaycastResult::Miss;
            }
        }

        // TODO: AABB normal, tangent, bitangent generation
        RaycastResult::Hit(Intersection {
            t: tmin,
            normal: Vec3::zero(),
            tangent: Vec3::zero(),
            bitangent: Vec3::zero(),
        })
    }

    // Construct AABB with inclusive boundaries
    fn from_triangles(triangles: &[Triangle]) -> AABB {
        let mut min = Vec3::new(std::f32::MAX);
        let mut max = Vec3::new(std::f32::MIN);

        for triangle in triangles {
            // Min
            if triangle.v0.x < min.x {
                min.x = triangle.v0.x;
            }
            if triangle.v1.x < min.x {
                min.x = triangle.v1.x;
            }
            if triangle.v2.x < min.x {
                min.x = triangle.v2.x;
            }

            if triangle.v0.y < min.y {
                min.y = triangle.v0.y;
            }
            if triangle.v1.y < min.y {
                min.y = triangle.v1.y;
            }
            if triangle.v2.y < min.y {
                min.y = triangle.v2.y;
            }

            if triangle.v0.z < min.z {
                min.z = triangle.v0.z;
            }
            if triangle.v1.z < min.z {
                min.z = triangle.v1.z;
            }
            if triangle.v2.z < min.z {
                min.z = triangle.v2.z;
            }

            // Max
            if triangle.v0.x > max.x {
                max.x = triangle.v0.x;
            }
            if triangle.v1.x > max.x {
                max.x = triangle.v1.x;
            }
            if triangle.v2.x > max.x {
                max.x = triangle.v2.x;
            }

            if triangle.v0.y > max.y {
                max.y = triangle.v0.y;
            }
            if triangle.v1.y > max.y {
                max.y = triangle.v1.y;
            }
            if triangle.v2.y > max.y {
                max.y = triangle.v2.y;
            }

            if triangle.v0.z > max.z {
                max.z = triangle.v0.z;
            }
            if triangle.v1.z > max.z {
                max.z = triangle.v1.z;
            }
            if triangle.v2.z > max.z {
                max.z = triangle.v2.z;
            }
        }

        AABB { min: min, max: max }
    }

    fn is_aabb_inside(&self, aabb: &AABB) -> Containment {
        let (a, b) = (
            self.is_point_inside(aabb.min),
            self.is_point_inside(aabb.max),
        );
        match (a, b) {
            (false, false) => Containment::None,
            (true, false) | (false, true) => Containment::Partial,
            (true, true) => Containment::Full,
        }
    }

    // Inclusive boundaries (according to AABB creation)
    fn is_point_inside(&self, p: Vec3) -> bool {
        let min = self.min;
        let max = self.max;
        if p.x <= max.x
            && p.y <= max.y
            && p.z <= max.z
            && p.x >= min.x
            && p.y >= min.y
            && p.z >= min.z
        {
            return true;
        }
        false
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
                    aabbs.push(aabb);
                }
            }
        }
        aabbs
    }
}

struct Material {
    emission: Vec3,
    brdf: Box<dyn brdf::BRDF>,
}

impl Material {
    fn new() -> Material {
        Material {
            emission: Vec3::zero(),
            brdf: Box::new(brdf::Lambertian::new(Vec3::zero())),
        }
    }

    fn shade(&self, scene: &Scene, ray: Ray, intersection: Intersection) -> Vec3 {
        if ray.depth == RAY_DEPTH_MAX {
            return Vec3::zero();
        }

        // NOTE: Incoming/outcoming RAY not RADIANCE
        let wi = intersection.to_shading(-ray.direction);
        let n = Vec3::from(0.0, 1.0, 0.0); // NOTE: Shading space
        let d = self.brdf.sample(wi, intersection.normal);
        let wo = intersection.to_shading(d);

        let next_ray = Ray {
            origin: ray.at(intersection.t) + (d * Vec3::new(0.0001)),
            direction: d,
            depth: ray.depth + 1,
        };

        let brdf = self.brdf.eval(wi, n, wo);
        let pdf = self.brdf.pdf(wi, n, wo);
        self.emission + brdf / pdf * scene.trace(next_ray)
    }
}

struct Octree {
    dimension: usize,
    aabbs: Vec<AABB>,
    triangles: Vec<Vec<Triangle>>,
    material: Material,
}

impl Octree {
    /// Octree construction of a single model
    fn new(dimension: usize, triangles: Vec<Triangle>, material: Material) -> Octree {
        let root_aabb = AABB::from_triangles(&triangles);

        let aabbs = root_aabb.subdivide(dimension);

        // FIXME: Couldnt this just be hashmap<Index, Triangle> instead?
        let mut aabb_triangles: Vec<Vec<Triangle>> = Vec::new();
        for _ in 0..dimension * dimension * dimension {
            aabb_triangles.push(Vec::new());
        }

        let mut did_not_fit_aabbs: u32 = 0;
        let mut partial_fits: u32 = 0;
        for triangle in triangles {
            let triangle_aabb = AABB::from_triangles(&vec![triangle]);
            let mut did_fit = false; // Did the triangle fit into a AABB?

            for (i, aabb) in aabbs.iter().enumerate() {
                match aabb.is_aabb_inside(&triangle_aabb) {
                    // Full containment of triangle in AABB, can move to next
                    Containment::Full => {
                        did_fit = true;
                        aabb_triangles[i].push(triangle);
                        break;
                    }
                    // Triangles may span several AABBs, must add to all
                    Containment::Partial => {
                        did_fit = true;
                        aabb_triangles[i].push(triangle);
                        partial_fits += 1;
                    }
                    Containment::None => continue,
                }
            }
            did_not_fit_aabbs += if !did_fit { 1 } else { 0 };
        }

        println!("# triangles not fitting to AABBs: {}", did_not_fit_aabbs);
        println!("# triangles partially fitting to a AABB: {}", partial_fits);

        Octree {
            dimension: dimension,
            aabbs: aabbs,
            triangles: aabb_triangles,
            material: material,
        }
    }

    fn intersects(&self, ray: &Ray) -> RaycastResult {
        let mut t0 = std::f32::MAX;
        let mut result = RaycastResult::Miss;
        // FIXME: Searching for ALL triangles hits in the AABBs measure AABB intersection t
        for (i, aabb) in self.aabbs.iter().enumerate() {
            match aabb.intersects(ray) {
                RaycastResult::Hit(_intersection) => {
                    for triangle in &self.triangles[i] {
                        let res = triangle.intersects(ray);
                        match res {
                            RaycastResult::Hit(intersection) => {
                                if intersection.t < t0 && intersection.t > std::f32::EPSILON {
                                    result = res;
                                    t0 = intersection.t;
                                }
                            }
                            RaycastResult::Miss => continue,
                        };
                    }
                }
                RaycastResult::Miss => continue,
            };
        }
        result
    }
}

// TODO: Add emitter/light list
struct Scene {
    octrees: Vec<Octree>,
}

impl Scene {
    fn new(path: &str) -> Scene {
        let start = std::time::Instant::now();
        println!("Starting to load scene {}", path);

        let mut octrees = Vec::new();

        let obj = tobj::load_obj(&Path::new(path));
        assert!(obj.is_ok());
        let (models, materials) = obj.unwrap();

        println!("# of models: {}", models.len());
        println!("# of materials: {}", materials.len());

        for (i, m) in models.iter().enumerate() {
            let mesh = &m.mesh;
            println!("model[{}].name = \'{}\'", i, m.name);
            println!("model[{}].mesh.material_id = {:?}", i, mesh.material_id);

            println!("model[{}].vertices: {}", i, mesh.positions.len() / 3);
            assert!(mesh.positions.len() % 3 == 0);

            println!("model[{}].indices: {}", i, mesh.indices.len());
            assert!(mesh.indices.len() % 3 == 0);

            println!("model[{}].triangles: {}", i, mesh.indices.len() / 3);

            let compute_normals = mesh.normals.is_empty();
            if compute_normals {
                println!("Computing normals!");
            }

            let mut triangles = Vec::new();
            for idxs in mesh.indices.chunks(3) {
                let mut v = [Vec3::zero(); 3];
                for i in 0..3 {
                    let idx = idxs[i] as usize;
                    let x = mesh.positions[3 * idx];
                    let y = mesh.positions[3 * idx + 1];
                    let z = mesh.positions[3 * idx + 2];
                    v[i] = Vec3 { x: x, y: y, z: z }
                }

                let mut n = [Vec3::zero(); 3];
                if compute_normals {
                    // FIXME: Must consider winding order of vertices when computing this
                    let normal = (v[0] - v[1]).cross(v[0] - v[2]).normalize();
                    n[0] = normal;
                    n[1] = n[0];
                    n[2] = n[0];
                } else {
                    for i in 0..3 {
                        let idx = idxs[i] as usize;
                        let x = mesh.normals[3 * idx];
                        let y = mesh.normals[3 * idx + 1];
                        let z = mesh.normals[3 * idx + 2];
                        n[i] = Vec3 { x: x, y: y, z: z }.normalize();
                    }
                }

                // Compute temp. bitangents
                let mut b = [Vec3::zero(); 3];
                b[0] = n[0].cross(v[1] - v[0]).normalize();
                b[1] = n[1].cross(v[2] - v[1]).normalize();
                b[2] = n[2].cross(v[0] - v[2]).normalize();

                // Compute tangents
                let mut t = [Vec3::zero(); 3];
                t[0] = b[0].cross(n[0]).normalize();
                t[1] = b[1].cross(n[1]).normalize();
                t[2] = b[2].cross(n[2]).normalize();

                // Recompute final bitangents
                b[0] = t[0].cross(n[0]).normalize();
                b[1] = t[1].cross(n[1]).normalize();
                b[2] = t[2].cross(n[2]).normalize();

                triangles.push(Triangle {
                    v0: v[0],
                    v1: v[1],
                    v2: v[2],
                    n0: n[0],
                    n1: n[1],
                    n2: n[2],
                    t0: t[0],
                    t1: t[1],
                    t2: t[2],
                    b0: b[0],
                    b1: b[1],
                    b2: b[2],
                });
            }

            if let Some(mid) = mesh.material_id {
                let mut material: Material = Material::new();

                let diffuse = Vec3 {
                    x: materials[mid].diffuse[0],
                    y: materials[mid].diffuse[1],
                    z: materials[mid].diffuse[2],
                };

                let brdf = brdf::Lambertian::new(diffuse);
                // let brdf = brdf::OrenNayar::new(diffuse, 1.0);
                material.brdf = Box::new(brdf);

                let emission = &materials[mid].unknown_param["Ke"];
                if !emission.is_empty() {
                    let strs = emission.split(" ").collect::<Vec<_>>();
                    if strs.len() == 3 {
                        let mut e = vec![0.0; 3];

                        for (i, s) in strs.iter().enumerate() {
                            if let Ok(num) = s.parse::<f32>() {
                                e[i] = num;
                            }
                        }

                        material.emission = Vec3 {
                            x: e[0],
                            y: e[1],
                            z: e[2],
                        };
                    }
                }

                // FIXME: Does not work in simple scenes for some unknown reason
                let dimension = 1;
                let octree = Octree::new(dimension, triangles, material);
                octrees.push(octree);
            }
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

    fn trace_background(&self, ray: Ray) -> Vec3 {
        let t = 0.5 * (ray.direction.y + 1.0);
        let white = Vec3::new(1.0);
        let light_blue = Vec3 {
            x: 0.5,
            y: 0.7,
            z: 1.0,
        };
        (1.0 - t) * white + t * light_blue
    }

    fn trace(&self, ray: Ray) -> Vec3 {
        let mut t0 = std::f32::MAX;
        let mut result = RaycastResult::Miss;
        let mut material = &Material::new();

        // TODO: Create a octree hierarchy instead of linear search?
        for octree in &self.octrees {
            let res = octree.intersects(&ray);
            match res {
                RaycastResult::Hit(intersection) => {
                    // NOTE: Why do we need to check against epsilon here? Already offsetting new rays in Material::shade()
                    if intersection.t < t0 && intersection.t > std::f32::EPSILON {
                        t0 = intersection.t;
                        material = &octree.material;
                        result = res;
                    }
                }
                RaycastResult::Miss => continue,
            };
        }

        match result {
            RaycastResult::Hit(intersection) => material.shade(&self, ray, intersection),
            RaycastResult::Miss => self.trace_background(ray),
        }
    }
}

enum Encoding {
    RGB { r: f32, g: f32, b: f32 }, // RGB with 32b float/channel
    ARGB(u32),                      // ARGB with 8 byte/channel
}

// Encodes the color and packing it in a Encoding enum
fn encode_color(encoding: Encoding, color: Vec3) -> Encoding {
    match encoding {
        Encoding::ARGB(_) => {
            // NOTE: Filter out NaNs is never equal to itself
            let x = if color.x != color.x { 0.0 } else { color.x };
            let y = if color.y != color.y { 0.0 } else { color.y };
            let z = if color.z != color.z { 0.0 } else { color.z };
            let ir = (x.clamp(0.0, 1.0).sqrt() * 255.99) as u32;
            let ig = (y.clamp(0.0, 1.0).sqrt() * 255.99) as u32;
            let ib = (z.clamp(0.0, 1.0).sqrt() * 255.99) as u32;
            let ia = 1_u32;
            let mut pixel = 0_u32;
            pixel += ia << 24;
            pixel += ir << 16;
            pixel += ig << 8;
            pixel += ib;
            Encoding::ARGB(pixel)
        }
        Encoding::RGB { r: _, g: _, b: _ } => Encoding::RGB {
            r: color.x,
            g: color.y,
            b: color.z,
        },
    }
}

// TODO: Spheres, define geometry mathematically, etc
// TODO: Use benchmarks to test performance of trace, shade, etc
fn main() {
    let mut screen_buffer: Vec<u32> = vec![0; WINDOW_WIDTH * WINDOW_HEIGHT];
    let mut screen_buffer_rgbf: Vec<Vec3> = vec![Vec3::zero(); WINDOW_WIDTH * WINDOW_HEIGHT];

    let mut window = Window::new(
        "Rusteray",
        WINDOW_WIDTH,
        WINDOW_HEIGHT,
        WindowOptions::default(),
    )
    .unwrap_or_else(|e| {
        panic!("{}", e);
    });

    let directory = std::env::current_dir().unwrap_or_default();
    let filepath = format!(
        "{}{}",
        directory.to_str().unwrap_or_default(),
        "/models/cornell_box/cornell_box.obj"
    );
    let scene = Arc::new(Scene::new(&filepath));

    // FIXME: From one axis does not work
    let from = Vec3 {
        x: 0.0,
        y: 1.0,
        z: 3.1,
    };

    let to = Vec3 {
        x: 0.0,
        y: 1.0,
        z: 0.0,
    };

    let camera = Camera::new(50.0, from, to);

    println!("Using {} threads in threadpool ... \n", num_cpus::get());
    let pool = ThreadPool::new(num_cpus::get());

    println!(
        "WINDOW: {}x{} \nRAY DEPTH MAX: {} \nSAMPLES PER PIXEL (SPP): {}\n",
        WINDOW_WIDTH, WINDOW_HEIGHT, RAY_DEPTH_MAX, SAMPLES_PER_PIXEL
    );

    let (tx, rx) = std::sync::mpsc::channel();

    let mut diff_sum = std::time::Duration::new(0, 0);
    let mut frames = 1;
    let mut rng = rand::thread_rng();
    let mut noise = BlueNoise::<Pcg64Mcg>::new(1.0, 1.0, 0.02);

    let start_time = std::time::Instant::now();
    while window.is_open() && !window.is_key_down(Key::Escape) {
        let start = std::time::Instant::now();
        let points = noise
            .with_seed(rng.gen())
            .take(SAMPLES_PER_PIXEL as usize)
            .collect::<Vec<_>>();

        for x in 0..WINDOW_WIDTH {
            for y in 0..WINDOW_HEIGHT {
                let tx = tx.clone();
                let scene = Arc::clone(&scene);
                let points = points.clone();
                pool.execute(move || {
                    let mut color = Vec3::zero();
                    for point in &points {
                        // Flipped variable t s.t y+ axis is up
                        let s = ((x as f32) + point.x) / (WINDOW_WIDTH as f32);
                        let t = ((y as f32) + point.y) / (WINDOW_HEIGHT as f32);
                        let ray = camera.ray(s, t);
                        color = color + scene.trace(ray);
                    }
                    let scale = 1.0 / (SAMPLES_PER_PIXEL as f32);
                    color = color * scale;
                    tx.send((x, y, color)).expect("Failed to send pixel!");
                });
            }
        }

        // Update and display progress bar - uncomment for small perf. inc.
        let work_units = WINDOW_WIDTH * WINDOW_HEIGHT;
        let progress_bar = indicatif::ProgressBar::new(work_units as u64);
        progress_bar.set_style(
            indicatif::ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise:.yellow}] [{wide_bar:.bold.green/blue}] {percent}% ({eta_precise:.yellow})")
                .progress_chars("=>-"),
        );

        for _ in 0..WINDOW_WIDTH * WINDOW_HEIGHT {
            let (x, y, pixel) = rx.recv().unwrap_or_default();

            let sum = ((screen_buffer_rgbf[y * WINDOW_WIDTH + x] * frames) + pixel)
                / ((frames + 1) as f32);
            screen_buffer_rgbf[y * WINDOW_WIDTH + x] = sum;

            // Encode as ARGB8
            screen_buffer[y * WINDOW_WIDTH + x] = match encode_color(Encoding::ARGB(0), sum) {
                Encoding::ARGB(encoded) => encoded,
                _ => 0,
            };

            progress_bar.inc(1);
        }

        progress_bar.finish();

        let end = std::time::Instant::now();
        let diff = end - start;

        let title = format!(
            "Rusteray - {}.{} s/frame",
            diff.as_secs(),
            diff.subsec_millis()
        );

        window.set_title(&title);

        frames += 1;
        diff_sum += diff;
        let info = format!(
            "- Frame #{}: {}.{} s/frame, {}.{} avg, {}.{} s \n",
            frames,
            diff.as_secs(),
            diff.subsec_millis(),
            (diff_sum / (frames as u32)).as_secs(),
            (diff_sum / (frames as u32)).subsec_millis(),
            (std::time::Instant::now() - start_time).as_secs(),
            (std::time::Instant::now() - start_time).subsec_millis()
        );
        println!("{}", info);

        window.update_with_buffer(&screen_buffer).unwrap();
    }
}
