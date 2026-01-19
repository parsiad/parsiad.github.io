# %% [raw]
# +++
# date = 2026-09-16
# title = "Much await about nothing: an intro to asyncio"
# +++

# %% tags=["no_cell"]
import asyncio
from collections.abc import Awaitable, Sequence
from typing import TypeVar

import httpx
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from _boilerplate import display_fig, init

T = TypeVar("T")

init()


def display_lanes(lanes: Sequence[Sequence[tuple[float, float, int]]]):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    tasks = [list(dict.fromkeys(task for _, _, task in lane)) for lane in lanes]
    row_count = sum(map(len, tasks))
    fig, ax = plt.subplots(figsize=(10, row_count), layout="constrained")
    fig.set_facecolor("white")
    ax.set_facecolor("white")
    cpu_positions, task_positions, task_labels = [], [], []
    row = 0
    for blocks, cpu_tasks in zip(lanes, tasks):
        cpu_positions.append(row + (len(cpu_tasks) - 1) / 2)
        for task in cpu_tasks:
            ax.hlines(row, 0, 1, color="0.8", zorder=0)
            for start, duration, task_number in blocks:
                if task_number == task:
                    ax.add_patch(
                        Rectangle(
                            (start, row - 0.25),
                            duration,
                            0.5,
                            facecolor=colors[(task - 1) % len(colors)],
                            edgecolor="white",
                        )
                    )
            task_positions.append(row)
            task_labels.append(f"Task {task}")
            row += 1
    ax.set_xlim(0, 1)
    ax.set_ylim(row_count - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    row = 0
    for i, cpu_tasks in enumerate(tasks):
        top = row - 0.25
        bottom = row + len(cpu_tasks) - 1 + 0.25
        transform = ax.get_yaxis_transform()  # x: axes fraction; y: row coordinates
        ax.plot(
            [-0.01, -0.03, -0.03, -0.01],
            [top, top, bottom, bottom],
            transform=transform,
            color="0.4",
            linewidth=1.5,
            clip_on=False,
        )
        ax.text(
            -0.05,
            (top + bottom) / 2,
            f"CPU{i + 1}",
            transform=transform,
            ha="right",
            va="center",
            fontsize=12,
            fontweight="bold",
        )
        row += len(cpu_tasks)
    ax.tick_params(axis="y", length=0, labelsize=12)
    right = ax.secondary_yaxis("right")
    right.set_yticks(task_positions, labels=task_labels)
    right.tick_params(axis="y", length=0, labelsize=12)
    for spine in (*ax.spines.values(), *right.spines.values()):
        spine.set_visible(False)
    display_fig(fig)


# %% [markdown]
# The purpose of this article is to serve as a practical introduction to [asyncio](https://docs.python.org/3/library/asyncio.html), a Python standard library package for concurrency.
# Concurrency (and, by extension, asyncio) is particularly useful when a program has multiple operations that spend time waiting for I/O.
# While one operation waits, another can make progress.
#
# ## Concurrency vs. parallelism
#
# People often confuse concurrency and parallelism.
# However, concurrency does not require simultaneous execution: operations can take turns making progress even on a single CPU.
# Let's define the two terms precisely:
#
# * **Concurrency**: making progress on multiple tasks without requiring one to finish before another.
# * **Parallelism**: executing multiple tasks at the same time.
#
# Examples of concurrency and parallelism are given below.
#
# ### Sequential (neither concurrent nor parallel)

# %% tags=["no_input"]
display_lanes(
    [
        [(0, 0.5, 1), (0.5, 0.5, 2)],
    ]
)

# %% [markdown]
# ### Concurrent, not parallel

# %% tags=["no_input"]
display_lanes(
    [
        [(0, 0.25, 1), (0.25, 0.25, 2), (0.5, 0.25, 1), (0.75, 0.25, 2)],
    ]
)

# %% [markdown]
# ### Concurrent and parallel

# %% tags=["no_input"]
display_lanes(
    [
        [(0, 0.25, 1), (0.25, 0.25, 2), (0.5, 0.25, 1), (0.75, 0.25, 2)],
        [(0, 0.25, 3), (0.25, 0.25, 4), (0.5, 0.25, 3), (0.75, 0.25, 4)],
    ]
)

# %% [markdown]
#
# ## An example involving multiple HTTP requests
#
# Suppose you have a list of webpages you want to fetch:

# %%
urls = [
    "https://www.example.com",
    "https://docs.python.org/3/library/asyncio.html",
]

# %% [markdown]
# You can fetch them sequentially with [httpx](https://www.python-httpx.org) as follows:


# %%
def fetch_page(client: httpx.Client, url: str) -> str:
    response = client.get(url)
    response.raise_for_status()  # Raise if the fetch failed
    return response.text


with httpx.Client() as client:
    results = [fetch_page(client, url) for url in urls]

# %% [markdown]
# This is a reasonable approach, especially when the number of pages is small.
# However, waiting for each response before starting the next request is wasteful, especially as the number of pages grows.
# Let's use concurrency instead:


# %%
async def async_fetch_page(client: httpx.AsyncClient, url: str) -> str:
    response = await client.get(url)
    response.raise_for_status()
    return response.text


client = httpx.AsyncClient()
results: list[str] = []
tasks: list[asyncio.Task[str]] = []
try:
    # Create one task per URL
    for url in urls:
        coro = async_fetch_page(client, url)
        task = asyncio.create_task(coro)
        tasks.append(task)
    # Await all tasks
    for task in tasks:
        result = await task
        results.append(result)
finally:
    await client.aclose()

# %% [markdown]
# There's a lot to unpack here if you are unfamiliar with asyncio, so let's proceed slowly:
#
# * `async def` defines a **coroutine function**. When invoked, it does not return the result (e.g., `response.text` in the above) directly; it returns a **coroutine** object.
# * Passing this coroutine to `asyncio.create_task` creates a **task** and schedules it to execute the work defined by the coroutine. The task is responsible for tracking progress and, if the work is finished, holding the result. Execution may not begin immediately, but you don't need to explicitly "start" the task.
# * `await` signifies that the outcome of an operation is needed before continuing.
#
# `await` can only be used within a coroutine function.
# In a Jupyter notebook, the kernel detects when a cell requires async execution (for example, because it contains a top-level `await`) and handles it accordingly.
# That makes the code above valid for execution in a Jupyter cell.
# Trying to execute it in the Python interpreter yields the following error:
# ```
#   File "<python-input-1>", line 18
#     result = await task
#              ^^^^^^^^^^
# SyntaxError: 'await' outside function
# ```
# To run it in the Python interpreter, wrap the top-level code in a coroutine function and call `asyncio.run(my_coroutine())`.
#
# We can simplify the concurrent requests code by using list comprehensions.
# Doing so does not introduce any new asyncio concepts.

# %%
client = httpx.AsyncClient()
try:
    results = [
        await task
        for task in [asyncio.create_task(async_fetch_page(client, url)) for url in urls]
    ]
finally:
    await client.aclose()

# %% [markdown]
# We can make the code more idiomatic by passing the coroutines directly to `asyncio.gather`:

# %%
client = httpx.AsyncClient()
try:
    results = await asyncio.gather(*[async_fetch_page(client, url) for url in urls])
finally:
    await client.aclose()

# %% [markdown]
# `asyncio.gather` wraps each coroutine in a task and returns a special awaitable object.
# This object is neither a coroutine nor a task (more on it later).
# Awaiting on it is conceptually similar but not identical to naively awaiting on all of the tasks it generates.
# For example, `asyncio.gather` propagates an early failure immediately (without cancelling sibling tasks).
#
# We can make the code even more idiomatic by using `async with`:

# %%
async with httpx.AsyncClient() as client:
    results = await asyncio.gather(*[async_fetch_page(client, url) for url in urls])

# %% [markdown]
#
# `async with` is analogous to `with` but the setup and cleanup methods are called with `await`:
#
# |              | Setup                | Cleanup                |
# | ------------ | -------------------- | ---------------------- |
# | `with`       | `__enter__()`        | `__exit__(...)`        |
# | `async with` | `await __aenter__()` | `await __aexit__(...)` |
#
# In the above, we have arrived at a concurrent version that is nearly as concise as the original synchronous version of the code, while allowing requests to make independent progress.
#
# ## Cooperative scheduling
#
# asyncio uses cooperative scheduling.
# That is, the event loop never interrupts a task to give another task a turn.
# Instead, the currently executing task keeps control until it suspends or finishes.
#
# An `await` expression is a possible suspension point, but it does not necessarily suspend.
# For example, directly awaiting a coroutine enters it immediately within the current task.
# This is made clear in the example below.


# %%
async def answer() -> int:
    return 42


async def announce() -> None:
    print("Hello, world")


task = asyncio.create_task(announce())
result = await answer()
print(result)
await task

# %% [markdown]
# With default task scheduling, creating the task schedules it but does not immediately execute it.
# Because `answer()` returns without suspending, awaiting it does not give the scheduled task a turn.
#
# To reverse the order of execution, we can use  `await asyncio.sleep(0)` to explicitly yield control:


# %%
task = asyncio.create_task(announce())
await asyncio.sleep(0)
result = await answer()
print(result)
await task

# %% [markdown]
# The rules are summarized below.
#
# | Await on | Does the current task suspend? |
# | --------- | ----------------------------------- |
# | Coroutine | If execution inside it suspends |
# | Task or future | If it is pending |
# | `asyncio.sleep(0)` | Always |
#
# Note that unlike `asyncio.sleep`, an ordinary `time.sleep` call blocks without yielding control.
#
# ## Cancellation
#
# To request the cancellation of a task, call `task.cancel()`.
# Note that this does not immediately stop the task: asyncio arranges for `CancelledError` to be raised inside the coroutine at its next opportunity.
# This is made clear in the example below.


# %%
async def worker() -> None:
    try:
        print("Start work")
        await asyncio.sleep(60)
        print("End work")
    finally:
        print("Start cleanup")
        await asyncio.sleep(1)  # Simulate cleanup
        print("End cleanup")


task = asyncio.create_task(worker())
await asyncio.sleep(0)
print("Request cancellation")
task.cancel()
try:
    await task
except asyncio.CancelledError:
    print("Task cancelled")


# %% [markdown]
# Often, several tasks form one operation: we want to wait for all of them, and if one fails, cancel the others and wait for their cleanup before continuing.
# This pattern is common enough that asyncio provides `TaskGroup` to handle it (available since Python 3.11).
#
# As an example, here it is applied to the earlier example involving concurrent requests:


# %%
async with httpx.AsyncClient() as client:
    async with asyncio.TaskGroup() as group:
        tasks = [group.create_task(async_fetch_page(client, url)) for url in urls]

    results = [task.result() for task in tasks]


# %% [markdown]
# `group.create_task(...)` schedules a coroutine as a task and registers it with the group.
# On leaving the group's block, `async with` awaits its `__aexit__` method,
# which waits for the group's tasks to finish.
#
# If all tasks succeed, we collect their results in the original URL order.
# If one fails with an ordinary exception, the group requests cancellation of the others,
# waits for their cleanup, and raises an `ExceptionGroup` containing the failures.
# In that case, the `results` assignment is not reached.
#
# Unlike the earlier `asyncio.gather` example, a failed request cannot leave sibling requests
# running after the HTTP client's context exits: the task group finishes first.
#
# ## Futures
#
# So far we have mainly dealt with tasks.
# In this section, we introduce **futures**.
# A plain future represents an eventual outcome supplied by other code.
# A task is a specialized future that also manages a coroutine's execution.
# This relationship is reflected in the class hierarchy:

# %%
issubclass(asyncio.Task, asyncio.Future)

# %% [markdown]
#
# We previously hinted at `asyncio.gather` returning a special awaitable object that is neither a coroutine nor a task.
# It turns out that this object is a future.
# To learn about how futures work, we introduce below a simplified implementation of `asyncio.gather`.


# %%
def my_gather(*awaitables: Awaitable[T]) -> asyncio.Future[list[T]]:
    loop = asyncio.get_running_loop()
    combined: asyncio.Future[list[T]] = loop.create_future()
    children: list[asyncio.Future[T]] = [asyncio.ensure_future(x) for x in awaitables]
    remaining = len(children)
    if remaining == 0:
        combined.set_result([])
        return combined

    def on_child_done(child: asyncio.Future[T]) -> None:
        nonlocal remaining
        remaining -= 1
        try:
            child.result()
        except BaseException as error:  # noqa: BLE001
            if not combined.done():
                combined.set_exception(error)
            return
        if remaining == 0 and not combined.done():
            results = [child.result() for child in children]
            combined.set_result(results)

    def on_combined_done(future: asyncio.Future[list[T]]) -> None:
        if future.cancelled():
            for child in children:
                child.cancel()

    combined.add_done_callback(on_combined_done)
    for child in children:
        child.add_done_callback(on_child_done)
    return combined


# %% [markdown]
#
# - `loop = asyncio.get_running_loop()` retrieves the event loop object. This object provides methods for scheduling work, creating tasks and futures, and registering callbacks for timers and I/O events.
# - `combined = loop.create_future()` creates a future associated to the event loop. This is the object the caller will eventually await.
# - `asyncio.ensure_future(...)` is used to normalize each awaitable input into a future:
#   - A future (including a task) is left unchanged;
#   - A coroutine is wrapped in a task;
#   - Other awaitables are wrapped in a coroutine and then in a task.
# - `on_child_done` is called when the child has succeeded, failed, or been cancelled.
#   - Calling `child.result()` on failure or cancellation raises an exception; we propagate that error with `combined.set_exception(...)`. Note that this also marks the future as done.
#   - If no exception is raised, we check to see if all children have finished and if the future is not done. This corresponds to all children having succeeded. In this case, we gather the results and assign them to the future with `combined.set_result(...)`.
# - `on_combined_done` is only used to propagate cancellations.
