#!/usr/bin/env python3

import ctypes as ct
import os
import subprocess
import sys
import time

from bcc import BPF

# Define the eBPF program
bpf_program = """
#include <uapi/linux/ptrace.h>

BPF_HASH(should_we_balance_count, u64, u64);

int trace_should_we_balance(struct pt_regs *ctx) {
    u64 retval = PT_REGS_RC(ctx);
    u64 key = 1, *count;

    if (retval == 1) {
        count = should_we_balance_count.lookup_or_try_init(&key, &retval);
        if (count) {
            (*count)++;
        }
    }

    return 0;
}
"""


def run_long_task(command):
    process = subprocess.Popen(command, shell=True, user="chin39", group="chin39")
    while process.poll() is None:
        time.sleep(0.001)  # Sleep for 1 ms to reduce CPU usage

    print("Long task completed.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: sudo python3 task_balancing_ebpf.py '<long_task_command>'")
        sys.exit(1)

    long_task_command = sys.argv[1]
    bpf = BPF(text=bpf_program)
    bpf.attach_kretprobe(event="should_we_balance", fn_name="trace_should_we_balance")

    print(f"Running long task: {long_task_command}")
    run_long_task(long_task_command)
    # Run the eBPF program for a specified duration
    # Detach the kprobe
    bpf.detach_kretprobe(event="should_we_balance")
    # Get the number of times the 'should_we_balance' function returned 1
    key = ct.c_ulonglong(1)
    count = bpf["should_we_balance_count"][key].value

    print(f"'should_we_balance' function returned 1 a total of {count} times.")
