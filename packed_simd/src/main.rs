use libc::{c_uint, syscall, SYS_getcpu};
use std::io;

use packed_simd::f32x8;
use rand::Rng;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use std::env;

const ARRAY_SIZE: usize = 1024;

fn main() {
    let args: Vec<String> = env::args().collect();
    let num_threads: usize = if args.len() > 1 {
        args[1].parse().expect("Invalid number of threads")
    } else {
        8
    };

    let mut rng = rand::thread_rng();
    let array1: Vec<f32> = (0..ARRAY_SIZE).map(|_| rng.gen::<f32>()).collect();
    let array2: Vec<f32> = (0..ARRAY_SIZE).map(|_| rng.gen::<f32>()).collect();

    let simd_thread_pool = ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .unwrap();

    let non_simd_thread_pool = ThreadPoolBuilder::new().num_threads(8).build().unwrap();

    for _ in 0..50000000 {
        let array1_clone = array1.clone();
        let array2_clone = array2.clone();
        non_simd_thread_pool.install(|| {
            elementwise_addition_parallel(&array1_clone, &array2_clone);
        });
        simd_thread_pool.install(|| {
            avx2_elementwise_addition_parallel(&array1, &array2);
        });
        // println!("Finished one round of calculations.");
    }
    print!("Finished all calculations.");
}

fn avx2_elementwise_addition_parallel(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "Arrays must have the same length.");

    let simd_chunks = a.len() / 8;
    let remainder = a.len() % 8;

    (0..simd_chunks).into_par_iter().for_each(|i| {
        let cpu_id = get_current_cpu_id().unwrap_or(0);
        let a_chunk = f32x8::from_slice_unaligned(&a[i * 8..(i + 1) * 8]);
        let b_chunk = f32x8::from_slice_unaligned(&b[i * 8..(i + 1) * 8]);
        let _sum_chunk = a_chunk + b_chunk;
        println!("SIMD CPU ID: {}", cpu_id);
    });

    if remainder > 0 {
        let offset = simd_chunks * 8;
        for _ in 0..remainder {
            let _sum = a[offset] + b[offset];
        }
    }
}

fn elementwise_addition_parallel(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "Arrays must have the same length.");

    a.par_iter().zip(b.par_iter()).for_each(|(a_elem, b_elem)| {
        let cpu_id = get_current_cpu_id().unwrap_or(0);
        let _sum = a_elem + b_elem;
        println!("Non-SIMD CPU ID: {}", cpu_id);
    });
}
fn get_current_cpu_id() -> io::Result<usize> {
    let mut cpu: c_uint = 0;
    let mut node: c_uint = 0;

    let result = unsafe {
        syscall(
            SYS_getcpu,
            &mut cpu as *mut c_uint,
            &mut node as *mut c_uint,
            std::ptr::null::<libc::c_void>() as *mut libc::c_void,
        )
    };

    if result == 0 {
        Ok(cpu as usize)
    } else {
        Err(io::Error::last_os_error())
    }
}
