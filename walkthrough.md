# KernelGYM 框架架构

## 概述

KernelGYM 是一个用于基准测试 CUDA/Triton 内核的分布式 GPU 内核评估服务。它提供基于 FastAPI 的服务器，接受评估请求、编排多阶段工作流（内核编译、正确性检查、性能计时），通过 Redis 支持的任务队列将工作分配给 GPU Worker，并在 CUDA 隔离的子进程池中执行任务。伴生 RL 训练层（`drkernel`）使用评估奖励通过 PPO 训练内核生成策略。

---

## 组件结构

```mermaid
graph TB
  subgraph Client_Layer["Client Layer"]
    Client["External Client / drkernel RL Trainer"]
  end

  subgraph API_Server["API Server - FastAPI"]
    API["FastAPI Server"]
    Models["Request/Response Models"]
  end

  subgraph Orchestration["Orchestration Layer"]
    WFC["WorkflowController"]
    KBWC["KernelBenchWorkflowController"]
    Sched["TaskManagerScheduler"]
  end

  subgraph Task_Mgmt["Task Management"]
    TM["TaskManager"]
    LB["WorkerLoadBalancer"]
    CRM["CodeRetryManager"]
    Redis[(Redis)]
  end

  subgraph Worker_Layer["Worker Layer"]
    WM["WorkerManager"]
    GPUw["GPUWorker"]
    Pool["SubprocessWorkerPool"]
    PW["PersistentWorker"]
  end

  subgraph Exec_Layer["Execution Layer"]
    TK["Toolkit - kernelbench"]
    BE["Backend"]
    GPU["GPU Device - CUDA"]
  end

  subgraph Monitoring["Monitoring"]
    WMon["WorkerMonitor"]
  end

  Client -->|"HTTP POST /evaluate"| API
  API --> KBWC
  KBWC -->|"submit/wait"| Sched
  Sched -->|"submit_task / get_task_result"| TM
  TM -->|"enqueue/dequeue"| Redis
  TM --> LB
  TM --> CRM
  GPUw -->|"get_next_task"| TM
  GPUw --> Pool
  Pool --> PW
  PW -->|"toolkit.evaluate"| TK
  TK -->|"compile/load/run"| BE
  BE --> GPU
  WMon -->|"health checks"| Redis
  WMon -->|"restart"| GPUw
```

---

## 时序图 1：端到端内核评估（关键路径）

**主要关键路径**：客户端提交内核进行评估，服务器编排两阶段工作流（内核评估 + 参考计时），结果最终返回给客户端。

```mermaid
sequenceDiagram
  autonumber
  participant C as 客户端
  participant API as FastAPI 服务器
  participant WC as KernelBenchWorkflowController
  participant S as TaskManagerScheduler
  participant TM as TaskManager
  participant R as Redis
  participant GW as GPUWorker
  participant SP as SubprocessWorkerPool
  participant PW as PersistentWorker
  participant TK as Toolkit
  participant BE as Backend
  participant GPU as GPU 设备

  Note over C, GPU: 阶段 1 - 请求提交与内核评估

  C->>+API: POST /evaluate (task_id, reference_code, kernel_code)
  API->>+WC: handle_request(input_data, scheduler)
  WC->>WC: validate_inputs(eval_task)
  WC->>WC: create_paired_tasks -> ref_task, kernel_task

  Note right of WC: 内核评估优先提交

  WC->>+S: submit(kernel_task_spec)
  S->>+TM: submit_task(kernel_payload)
  TM->>R: HSET task:id status=pending
  TM->>R: LPUSH queue:priority:normal task_id
  TM-->>-S: task_id
  S-->>-WC: kernel_task_id

  WC->>+S: wait(kernel_task_id)

  Note over GW, GPU: Worker 处理循环（异步）

  GW->>+TM: get_next_task(worker_id)
  TM->>R: RPOP queue:worker:id or queue:priority:*
  TM->>R: HSET task:id status=processing
  TM-->>-GW: task_data

  GW->>+SP: execute_task(task_data, timeout)
  SP->>SP: _get_idle_worker -> PersistentWorker
  SP->>+PW: execute_task via run_in_executor

  Note right of PW: 隔离子进程，拥有独立 CUDA 上下文

  PW->>+TK: toolkit.evaluate(task_data, backend)
  TK->>+BE: compile(kernel_code)
  BE->>GPU: Load and compile kernel
  BE-->>-TK: compiled artifact

  TK->>+BE: run - correctness check
  BE->>GPU: Execute kernel, compare outputs
  BE-->>-TK: correctness result

  TK->>+BE: run - performance timing
  BE->>GPU: Execute kernel N trials
  BE-->>-TK: kernel_runtime

  TK-->>-PW: KernelExecResult
  PW-->>-SP: success result

  Note right of SP: 若发生 CUDA 错误：worker_exiting=true，Pool 自动重启

  SP-->>-GW: result_data

  GW->>+TM: complete_task(task_id, result_dict)
  TM->>R: HSET result:task_id, status=completed
  TM-->>-GW: done

  Note over S: Scheduler 轮询 Redis 获取结果
  S->>TM: get_task_result(kernel_task_id)
  TM->>R: HGETALL result:task_id
  TM-->>S: kernel_result_dict
  S-->>-WC: kernel_result_dict

  Note over C, GPU: 阶段 2 - 参考计时（仅当内核通过编译和正确性检查后执行）

  WC->>WC: 检查 compiled AND correctness = true

  alt 存在缓存的参考运行时
    WC->>WC: 使用缓存的 reference_runtime
  else 无缓存 - 提交参考计时任务
    WC->>+S: submit(ref_task_spec)
    S->>TM: submit_task(ref_payload)
    TM->>R: LPUSH queue:priority:normal
    S-->>-WC: ref_task_id

    WC->>+S: wait(ref_task_id)
    Note over GW, GPU: Worker 拾取参考计时任务（流程同阶段 1）
    GW->>TM: get_next_task
    GW->>SP: execute_task -> reference_runtime
    GW->>TM: complete_task(ref_task_id, ref_result)
    S->>TM: get_task_result(ref_task_id)
    S-->>-WC: ref_result_dict
  end

  Note over C, GPU: 阶段 3 - 结果聚合

  WC->>WC: combine_results -> EvaluationResult with speedup
  WC->>WC: _persist_result to eval_results.jsonl
  WC-->>-API: final EvaluationResult dict

  API->>TM: complete_task(task_id, result)
  API-->>-C: EvaluationResponse (status, compiled, correctness, speedup)
```

---

## 时序图 2：Worker 生命周期与健康监控

Worker 注册、心跳循环、CUDA 错误隔离，以及监控器触发的自动重启流程。

```mermaid
sequenceDiagram
  autonumber
  participant WM as WorkerManager
  participant GW as GPUWorker
  participant API as API 服务器
  participant TM as TaskManager
  participant R as Redis
  participant SP as SubprocessWorkerPool
  participant PW as PersistentWorker
  participant Mon as WorkerMonitor

  Note over WM, Mon: Worker 启动

  WM->>GW: create GPUWorker(worker_id, cuda:N, redis)
  GW->>R: HSET worker:id online=initializing
  GW->>API: POST /node/allocate -> node_id
  GW->>API: POST /worker/register (worker_id, device, node_id)
  API->>+TM: register_worker(worker_id, device)
  TM->>R: HSET worker:id status=online
  TM->>TM: WorkerLoadBalancer.register_worker()
  TM-->>-API: success
  GW->>GW: _initialize_gpu() 通过 nvidia-smi（主进程不初始化 CUDA）

  GW->>+SP: SubprocessWorkerPool(device_id, pool_size)
  SP->>+PW: PersistentWorker(device_id) via mp.spawn
  PW->>PW: torch.cuda.init(), set_device(), warmup
  PW-->>-SP: READY with init_time
  SP-->>-GW: pool initialized

  Note over WM, Mon: 心跳循环（每 10 秒）

  loop 每 10 秒
    GW->>API: POST /worker/heartbeat
    API->>TM: update_worker_heartbeat(worker_id)
    TM->>R: HSET worker:id last_heartbeat, status=online
    GW->>R: HSET worker:id online=true, stats
    GW->>R: EXPIRE worker:id 120s
  end

  Note over WM, Mon: CUDA 错误隔离与自动重启

  PW->>PW: 任务触发 CUDA 错误
  PW->>SP: result (success=false, worker_exiting=true)
  SP->>SP: 标记 PersistentWorker 为不可用
  SP->>SP: _restart_worker(worker)
  SP->>PW: shutdown(timeout=5)
  SP->>SP: sleep(2s) 等待 GPU 资源释放
  SP->>+PW: new PersistentWorker(same device_id)
  PW->>PW: fresh torch.cuda.init()
  PW-->>-SP: READY

  Note over WM, Mon: 监控器触发重启（崩溃恢复）

  Mon->>R: KEYS worker:* 扫描所有 Worker
  Mon->>R: HGETALL worker:id 检查心跳时间

  alt 心跳超时
    Mon->>Mon: 将 worker_id 加入重启队列
    Mon->>Mon: _kill_worker_process（SIGTERM 后升级 SIGKILL）
    Mon->>Mon: _reset_gpu_device（nvidia-smi --gpu-reset）
    Mon->>Mon: 启动新的 single_worker 进程
    Mon->>R: HSET worker_process:id pid, start_time
  end

  alt 持久化模式 - 心跳键缺失
    Mon->>R: SMEMBERS expected_workers
    Mon->>Mon: 与现有 Worker 键对比
    Mon->>Mon: 重启所有缺失的期望 Worker
  end
```

---

## 时序图 3：子进程池内部流程

[SubprocessWorkerPool](file:///home/robomaster/Research/KernelGYM/kernelgym/worker/subprocess_pool.py#298-627) 与 [PersistentWorker](file:///home/robomaster/Research/KernelGYM/kernelgym/worker/subprocess_pool.py#100-296) 的内部流程，包括通过多进程队列进行的进程间通信（IPC）。

```mermaid
sequenceDiagram
  autonumber
  participant GW as GPUWorker
  participant SP as SubprocessWorkerPool
  participant Lock as asyncio.Lock
  participant PW as PersistentWorker
  participant TQ as task_queue（mp.Queue）
  participant RQ as result_queue（mp.Queue）
  participant Loopw as worker_loop（子进程）
  participant TK as Toolkit.evaluate
  participant BE as Backend
  participant GPU as GPU

  GW->>+SP: execute_task(task_data, timeout=35, max_retries=2)

  loop retry_count <= max_retries
    SP->>+Lock: 获取锁
    SP->>SP: 过滤 idle_workers（移除已死亡的）
    SP->>SP: 取出 idle_worker 得到 PW
    SP->>SP: 将 PW 移至 busy_workers
    Lock-->>-SP: 释放锁

    SP->>+PW: execute_task via run_in_executor

    PW->>TQ: put(task_data)

    Note right of Loopw: 子进程从队列中拾取任务

    Loopw->>TQ: get() task_data
    Loopw->>Loopw: 从缓存解析 toolkit 和 backend
    Loopw->>+TK: toolkit.evaluate(task_data, backend)
    TK->>+BE: compile then load then run
    BE->>GPU: CUDA execution
    GPU-->>BE: results
    BE-->>-TK: exec_result
    TK-->>-Loopw: result_dict

    Loopw->>Loopw: _aggressive_gpu_cleanup(device_id)
    Loopw->>RQ: put(success=true, result=result_dict)

    PW->>RQ: get(timeout) result
    PW->>PW: tasks_processed++

    alt tasks_processed >= max_tasks_per_worker
      PW->>PW: is_alive_flag = false（标记为待重启）
    end

    PW-->>-SP: result

    SP->>SP: total_tasks_processed++

    alt Worker 需要重启（不可用）
      SP->>SP: _restart_worker(PW)
    end

    SP->>+Lock: 获取锁
    SP->>SP: 从 busy_workers 移除 PW
    SP->>SP: 若 PW 存活则加回 idle_workers
    Lock-->>-SP: 释放锁

    SP-->>-GW: result_data
  end
```

---

## 时序图 4：批量评估与工作流提交

备用 API 入口：批量评估和通用工作流提交，支持 Redis 结果缓存。

```mermaid
sequenceDiagram
  autonumber
  participant C as 客户端
  participant API as FastAPI 服务器
  participant TM as TaskManager
  participant WC as KernelBenchWorkflowController
  participant S as TaskManagerScheduler
  participant R as Redis

  Note over C, R: 批量评估 /evaluate/batch

  C->>+API: POST /evaluate/batch (batch_id, tasks 列表)

  loop 对批次中每个任务
    API->>API: _execute_workflow(workflow, payload, task_id)

    alt Redis 中已有缓存结果
      API->>TM: get_task_result(task_id)
      TM->>R: HGETALL result:task_id
      R-->>TM: 缓存结果
      TM-->>API: 直接返回（跳过重新评估）
    else 无缓存或 force_refresh
      API->>S: TaskManagerScheduler(task_mgr)
      API->>WC: handle_request(payload, scheduler)
      Note right of WC: 完整评估流程（见时序图 1）
      WC-->>API: EvaluationResult
      API->>TM: complete_task(task_id, result)
    end
  end

  API-->>-C: BatchEvaluationResponse (total, completed, failed, results)

  Note over C, R: 通用工作流提交 /workflow/submit

  C->>+API: POST /workflow/submit (workflow, payload, task_id)
  API->>API: 通过 get_workflow_controller(name) 解析控制器
  API->>+WC: handle_request(payload, scheduler)
  WC-->>-API: result
  API->>TM: complete_task(task_id, result)
  API-->>-C: WorkflowResponse (task_id, status, result)
```

---

## 关键架构决策

| 决策 | 设计依据 |
|------|----------|
| **子进程 CUDA 隔离**（spawn 模式） | 主 GPUWorker 进程永不初始化 CUDA。所有 GPU 工作在 [PersistentWorker](file:///home/robomaster/Research/KernelGYM/kernelgym/worker/subprocess_pool.py#100-296) 子进程中执行。CUDA 错误仅终止子进程，主 Worker 自动重启新子进程。 |
| **持久化 Worker 池** vs 每任务 spawn | 通过复用已初始化的子进程，消除每任务约 2.5s 的启动开销。Worker 在处理 `max_tasks_per_worker` 任务后自动回收以防止显存累积。 |
| **Redis 作为中央状态存储** | 任务队列、Worker 注册表、心跳、结果均存储于 Redis，使 API 服务器与 GPU Worker 跨节点解耦。 |
| **两阶段工作流**（内核 → 参考计时） | 仅当内核通过编译+正确性检查后才提交参考计时任务，节省失败内核的 GPU 时间。参考计时结果可缓存复用。 |
| **WorkerMonitor** 作为独立进程 | 独立运行，周期性扫描 Redis 检测崩溃/死亡 Worker 并重启。支持持久化模式，通过期望 Worker 集合管理集群部署。 |
| **优先级队列**（high/normal/low） | TaskManager 支持基于优先级的调度及每 Worker 专属队列以实现亲和性路由。 |

---

## 数据流概览

```
客户端请求
    |
    v
FastAPI /evaluate --> KernelBenchWorkflowController
    |                        |
    |                  +-----+-----+
    |                  | 阶段 1    | 阶段 2
    |                  v           v
    |          内核评估 Eval   参考计时 Ref
    |          TaskSpec         TaskSpec
    |              |              |
    |              v              v
    |     TaskManagerScheduler.submit()
    |              |
    |              v
    |     TaskManager --> Redis 任务队列
    |              |
    |          +---+
    |          v
    |     GPUWorker.get_next_task()
    |          |
    |          v
    |     SubprocessWorkerPool
    |          |
    |          v
    |     PersistentWorker（子进程）
    |          |
    |     +----+----+
    |     v         v
    |  Toolkit   Backend
    |     |         |
    |     +----+----+
    |          v
    |     GPU 执行
    |          |
    |          v
    |     结果写入 Redis
    |          |
    |          v
    |     Scheduler.wait() 轮询 Redis
    |          |
    |          v
    |     WorkflowController 聚合结果
    |          |
    |          v
    +---> EvaluationResponse --> 客户端
```
