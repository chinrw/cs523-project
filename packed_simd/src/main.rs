use itertools::Itertools;
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
    // mix of simd and non-simd
    let simd_thread = 64;
    let gcd_thread = 64;

    let cpu_test_ids = "0-31";
    let gcd_test_ids = "0-15";

    let simd_stress_ng_command =
        format!("taskset -c {cpu_test_ids} stress-ng --cpu {simd_thread} --cpu-method fft --metrics-brief --cpu-ops 3000000");
    let gcd_stress_ng_command =
        format!("taskset -c {gcd_test_ids} stress-ng --cpu {gcd_thread} --cpu-method gcd --metrics-brief --cpu-ops 2000000");

    println!("Running simd stress-ng command: {}", simd_stress_ng_command);
    println!("Running gcd stress-ng command: {}", gcd_stress_ng_command);

    let simd_task = tokio::spawn(async move {
        run_command_with_priority(&simd_stress_ng_command, 15);
    });

    let gcd_task = tokio::spawn(async move {
        run_command_with_priority(&gcd_stress_ng_command, 0);
    });

    tokio::try_join!(simd_task, gcd_task).unwrap();
}

async fn stress_ng_bench_cpu_affinity() {
    // mix of simd and non-simd
    let simd_thread = 64;
    let gcd_thread = 64;

    let cpu_test_ids = "0-31";
    let gcd_test_ids = "16-31";

    let simd_stress_ng_command =
        format!("taskset -c {cpu_test_ids} stress-ng --cpu {simd_thread} --cpu-method fft --metrics-brief --cpu-ops 3000000");
    let gcd_stress_ng_command =
        format!("taskset -c {gcd_test_ids} stress-ng --cpu {gcd_thread} --cpu-method gcd --metrics-brief --cpu-ops 2000000");

    println!("Running simd stress-ng command: {}", simd_stress_ng_command);
    println!("Running gcd stress-ng command: {}", gcd_stress_ng_command);

    let simd_task = tokio::spawn(async move {
        run_command_with_priority(&simd_stress_ng_command, 15);
    });

    let gcd_task = tokio::spawn(async move {
        run_command_with_priority(&gcd_stress_ng_command, 0);
    });

    tokio::try_join!(simd_task, gcd_task).unwrap();
    println!("All stress-ng instances completed.");
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let bench_type = match args.get(1) {
        Some(arg) => arg.to_string(),
        None => String::from("bench"),
    };

    let rt = Runtime::new().unwrap();
    if bench_type == "stress_cpu_affinity" {
        rt.block_on(stress_ng_bench_cpu_affinity())
    } else if bench_type == "stress" {
        rt.block_on(stress_ng_bench_cpu());

        // stress_ng_bench_cpu();
    } else if bench_type == "simd" {
        hand_made_simd(&args);
    }
}
