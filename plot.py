import matplotlib.pyplot as plt


def plot_task_balancing_data():
    y = [21987, 17966]
    x = ["6.2-origin", "class-patched"]

    fig, ax = plt.subplots()
    bars = ax.bar(x, y)

    ax.set_xlabel("kernel")
    ax.set_ylabel("load_balance function")
    ax.set_title("need_active_balance count")

    ax.set_xticks(x)
    # ax.set_xticklabels(
    #     [f'{"*" if cpu in smt_threads else ""}{cpu}' for cpu in x],
    #     rotation=90,
    #     ha="right",
    # )  # Rotate x-axis labels
    plt.tight_layout()  # Adjust spacing
    plt.savefig("task_balancing_bar_chart_cores.png")


plot_task_balancing_data()
