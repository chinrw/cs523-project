use itertools::Itertools;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokio::runtime::Runtime;

use rand::Rng;
use std::env;
mod array_bench;
mod helper_functions;
use array_bench::*;
use helper_functions::*;

fn hand_made_simd(args: &[String]) {
    let num_interations = args
        .get(3)
        .and_then(|arg| arg.parse().ok())
        .unwrap_or(100000);
    let mut rng = rand::thread_rng();
    let array1: Vec<f32> = (0..ARRAY_SIZE).map(|_| rng.gen::<f32>()).collect();
    let array2: Vec<f32> = (0..ARRAY_SIZE).map(|_| rng.gen::<f32>()).collect();

    let simd_cpu_ids = Arc::new(Mutex::new(HashMap::new()));
    let non_simd_cpu_ids = Arc::new(Mutex::new(HashMap::new()));

    let num_threads: usize = args.get(2).and_then(|arg| arg.parse().ok()).unwrap_or(12);
    let simd_thread_pool = ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .unwrap();

    let num_threads_non_simd = args.get(2).and_then(|arg| arg.parse().ok()).unwrap_or(12);
    let non_simd_thread_pool = ThreadPoolBuilder::new()
        .num_threads(num_threads_non_simd)
        .build()
        .unwrap();

    for _ in 0..num_interations {
        let array1_clone = array1.clone();
        let array2_clone = array2.clone();
        let non_simd_cpu_ids_clone = non_simd_cpu_ids.clone();

        non_simd_thread_pool.install(|| {
            elementwise_addition_parallel(&array1_clone, &array2_clone, non_simd_cpu_ids_clone);
        });
        let simd_cpu_ids_clone = simd_cpu_ids.clone();

        simd_thread_pool.install(|| {
            avx2_elementwise_addition_parallel(&array1, &array2, simd_cpu_ids_clone);
        });
        // println!("Finished one round of calculations.");
    }

    let sorted_simd_cpu_id_frequencies: Vec<(usize, usize)> = simd_cpu_ids
        .lock()
        .unwrap()
        .iter()
        .map(|(&k, &v)| (k, v))
        .sorted_by(|b, a| a.1.cmp(&b.1))
        .collect();

    println!(
        "SIMD CPU ID frequencies: {:?}",
        sorted_simd_cpu_id_frequencies
    );

    let sorted_non_simd_cpu_id_frequencies: Vec<(usize, usize)> = non_simd_cpu_ids
        .lock()
        .unwrap()
        .iter()
        .map(|(&k, &v)| (k, v))
        .sorted_by(|b, a| a.1.cmp(&b.1))
        .collect();

    println!(
        "Non-SIMD CPU ID frequencies: {:?}",
        sorted_non_simd_cpu_id_frequencies
    );

    create_chart(
        "simd_cpu_id_frequencies.png",
        "SIMD CPU ID Frequencies",
        &sorted_simd_cpu_id_frequencies,
    )
    .unwrap();

    create_chart(
        "non_simd_cpu_id_frequencies.png",
        "Non-SIMD CPU ID Frequencies",
        &sorted_non_simd_cpu_id_frequencies,
    )
    .unwrap();

    println!("Finished all calculations.");
}

async fn stress_ng_bench_cpu() {
    let simd_thread = 2;
    let gcd_thread = 2;
    // Run two thread due to hyperthreading
    let simd_stress_ng_command =
        format!("stress-ng --cpu {simd_thread} --cpu-method fft --metrics-brief --cpu-ops 50000");
    let gcd_stress_ng_command =
        format!("stress-ng --cpu {gcd_thread} --cpu-method gcd --metrics-brief --cpu-ops 50000");
    let simd_task = tokio::spawn(async move {
        run_command(&simd_stress_ng_command);
    });

    let gcd_task = tokio::spawn(async move {
        run_command(&gcd_stress_ng_command);
    });

    tokio::try_join!(simd_task, gcd_task).unwrap();
}

fn stress_ng_bench_cpu_affinity() {
    // Run two thread due to hyperthreading
    let simd_stress_ng_command = "stress-ng --cpu 1 --cpu-method fft --metrics-brief --cpu-ops 50000";
    let gcd_stress_ng_command = "stress-ng --cpu 1 --cpu-method gcd --metrics-brief --cpu-ops 50000";

    let gcd_test_ids: Vec<usize> = (15..=16).collect(); // Adjust these values based on your CPU configuration
    let cpu_test_ids: Vec<usize> = (0..=1).collect(); // Adjust these values based on your CPU configuration

    let test_configs: Vec<(&str, usize)> = cpu_test_ids
        .iter()
        .map(|&id| (simd_stress_ng_command, id))
        .chain(gcd_test_ids.iter().map(|&id| (gcd_stress_ng_command, id)))
        .collect();

    // Create a new thread pool with the total number of threads required for both tests
    let thread_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(test_configs.len())
        .build()
        .unwrap();

    // Use the custom thread pool to run the stress-ng commands on each CPU ID
    thread_pool.install(|| {
        test_configs.par_iter().for_each(|&(command, cpu_id)| {
            println!("Running stress-ng on CPU core {}: {}", cpu_id, command);
            run_taskset(command, cpu_id);
        });
    });
    println!("All stress-ng instances completed.");
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let bench_type = match args.get(1) {
        Some(arg) => arg.to_string(),
        None => String::from("bench"),
    };

    if bench_type == "stress_cpu_affinity" {
        stress_ng_bench_cpu_affinity();
    } else if bench_type == "stress" {
        let rt = Runtime::new().unwrap();
        rt.block_on(stress_ng_bench_cpu());

        // stress_ng_bench_cpu();
    } else if bench_type == "simd" {
        hand_made_simd(&args);
    }
}
