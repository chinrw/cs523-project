#include <linux/ptrace.h>
#include <linux/sched.h>
#include <linux/bpf.h>


struct data_t {
    u32 cpu;
    u32 pid;
    u64 runtime;
    u64 vruntime;
};

BPF_PERF_OUTPUT(events);

int trace_pick_next_task_fair(struct pt_regs *ctx, struct rq *rq) {
    struct data_t data = {};
    struct task_struct *next = (struct task_struct *)bpf_get_current_task();

    data.cpu = bpf_get_smp_processor_id();
    data.pid = next->pid;
    data.runtime = next->se.sum_exec_runtime;
    data.vruntime = next->se.vruntime;

    events.perf_submit(ctx, &data, sizeof(data));
    return 0;
}

