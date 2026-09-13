# Local Multi-Agent AI Platform

## 1. Project Vision

Build a desktop/local platform that allows users to download or provide a local Hugging Face model and run it locally through a selectable inference backend.

The main purpose of the platform is NOT simply to be another local LLM chat UI.

The core idea is to investigate and enable **multiple instances of the same relatively small/cheap model cooperating with each other**.

For example:

```text
User Prompt
     |
     +-------------------+
     |                   |
     v                   v
 Model Instance A    Model Instance B
     |                   |
     |                   |
     +---------+---------+
               |
               v
        Orchestrator / Judge
               |
               v
          Final Answer
```

The hypothesis is:

> Multiple independent instances of a small model, combined with structured collaboration, criticism, verification, or voting, may outperform a single instance of the same model and potentially compete with substantially larger models at a lower computational cost.

The platform should make experimenting with this idea extremely easy.

---

# 2. First Target Model

The first model to test is:

`google/gemma-4-E4B`

Hugging Face:

`https://huggingface.co/google/gemma-4-E4B`

However, DO NOT hard-code Gemma 4 E4B into the architecture.

Gemma 4 E4B is simply the first model used to validate the platform.

The architecture must be designed around supporting many Hugging Face-compatible models.

---

# 3. User Experience

The user should not need to understand model-serving infrastructure.

The intended workflow is:

```text
Install application
       |
       v
Choose inference backend
       |
       +---- Ollama
       |
       +---- AirLLM
       |
       +---- Auto
       |
       v
Launch application
       |
       v
Add/import model
       |
       v
Platform detects model
       |
       v
Configure number of agents
       |
       v
Choose collaboration strategy
       |
       v
Run
```

The user should be able to provide either:

* a local model directory
* a supported Hugging Face model identifier
* potentially a model file/format supported by the selected backend

The platform should inspect the model and determine compatibility rather than blindly assuming every model works.

---

# 4. Inference Backends

The user should be able to select an inference backend during installation.

Initial supported backends:

## Ollama

Use Ollama when appropriate for easy local model execution and good performance.

Advantages:

* Simple installation
* User-friendly
* Good local inference experience
* GPU acceleration
* Model management
* Suitable for models supported by Ollama

The application should communicate with Ollama through an internal adapter rather than coupling the rest of the application directly to Ollama APIs.

---

## AirLLM

AirLLM should be supported as an alternative backend.

Primary reason:

AirLLM can make it practical to run models that would otherwise be difficult to fit entirely into available GPU VRAM by using layer-wise/offloading techniques.

This makes AirLLM particularly useful for users with limited GPU memory.

However:

DO NOT assume AirLLM supports literally every Hugging Face model.

The platform must detect compatibility and report failures clearly.

---

## Auto

Eventually provide:

`Auto`

The platform determines the most appropriate backend based on:

* model architecture
* model format
* model size
* available GPU
* available VRAM
* system RAM
* operating system
* backend compatibility
* expected performance

Example:

```text
Gemma 4 E4B
        |
        v
Model Inspector
        |
        v
Fits comfortably in GPU memory
        |
        v
Recommend Ollama
```

Another example:

```text
Large model
        |
        v
Insufficient GPU VRAM
        |
        v
Recommend AirLLM
```

The user must still be able to override the automatic recommendation.

---

# 5. Runtime Abstraction

This is one of the most important architectural requirements.

The application MUST NOT make the orchestrator depend directly on Ollama or AirLLM.

Create a runtime abstraction.

Conceptually:

```text
Runtime
├── load_model()
├── unload_model()
├── generate()
├── stream()
├── get_model_info()
├── get_status()
├── get_capabilities()
└── health_check()
```

Then implement:

```text
Runtime
├── OllamaRuntime
└── AirLLMRuntime
```

Future runtimes should be easy to add.

Possible future backends:

* llama.cpp
* vLLM
* SGLang
* Transformers
* MLX
* other local inference engines

The rest of the application should not care which runtime is being used.

---

# 6. Agent Abstraction

An "agent" is NOT necessarily a separate copy of the model weights.

This distinction is important.

Two agents may use the same loaded model/runtime while maintaining independent conversational/inference state.

Conceptually:

```text
                 Runtime
                    |
          +---------+---------+
          |                   |
          v                   v
      Agent A              Agent B
      State A              State B
          |                   |
          +---------+---------+
                    |
                    v
               Orchestrator
```

The system should avoid unnecessarily loading duplicate model weights when the runtime can safely share them.

The agent manager should therefore distinguish between:

* model instance
* inference session
* agent
* conversation state
* KV/cache state
* runtime process

---

# 7. Multi-Agent Strategies

The platform should support multiple strategies.

Initial strategies:

## Single

One model instance produces the answer.

```text
Prompt
  |
  v
Agent
  |
  v
Answer
```

This provides the baseline.

---

## Independent

Multiple agents independently solve the same problem.

```text
             Prompt
            /     \
           v       v
       Agent A   Agent B
           |       |
           v       v
        Answer A Answer B
```

The user can compare the outputs.

---

## Debate

Multiple agents produce solutions and critique each other.

Example:

```text
Prompt
  |
  +----------+
  |          |
  v          v
Agent A    Agent B
  |          |
  +----+-----+
       |
       v
Critique
       |
       v
Final response
```

The exact number of rounds should be configurable.

---

## Solver → Critic

One agent generates a solution.

Another agent reviews it for:

* correctness
* hallucinations
* missing requirements
* bugs
* edge cases
* logical errors

The first agent may then revise its answer.

```text
Solver
  |
  v
Solution
  |
  v
Critic
  |
  v
Feedback
  |
  v
Solver revision
```

---

## Majority Vote

Multiple independent agents answer.

The system chooses the answer based on agreement.

This is particularly useful for:

* multiple-choice questions
* classification
* deterministic reasoning problems
* benchmark experiments

---

## Judge

Multiple agents generate answers.

A separate judge evaluates them.

The judge should ideally be configurable.

The judge can be:

* another instance of the same model
* a different local model
* eventually a remote model

Do not hard-code the judge to a specific model.

---

# 8. Benchmarking

Benchmarking should be a first-class feature.

The platform should allow users to compare:

```text
1 Agent
vs
2 Agents
vs
3 Agents
vs
4 Agents
```

and compare strategies.

Metrics should include:

* accuracy
* task completion
* latency
* tokens generated
* tokens/second
* total inference time
* memory usage
* VRAM usage where available
* number of model calls
* number of reasoning/critique rounds

The system should preserve benchmark results so configurations can be compared.

Example:

```text
                    Accuracy    Time    Memory
------------------------------------------------
1× E4B               72%       12s     6GB
2× E4B Independent   76%       19s     7GB
2× E4B Debate        81%       31s     8GB
3× E4B Judge         84%       44s     9GB
```

These numbers are examples only.

NEVER fabricate benchmark results.

---

# 9. Experimentation

The platform should make it easy to test the central hypothesis:

> Can multiple small models outperform a single larger model?

Eventually the benchmark interface should support:

```text
Model A
1 instance

Model A
2 instances + debate

Model A
3 instances + judge

Model B
1 instance

Model C
1 instance
```

Then produce a comparison.

Important:

The platform must distinguish between:

* raw model capability
* orchestration improvement
* additional compute
* latency
* memory consumption

A multi-agent configuration should not be declared "better" merely because it produces a longer answer.

---

# 10. Model Inspector

When a model is added, inspect available metadata.

Potential information:

* model architecture
* parameter count if available
* model format
* quantization
* context length
* supported modalities
* tokenizer
* required files
* runtime compatibility
* estimated memory requirements

Example:

```text
Model
Gemma 4 E4B

Architecture
Gemma

Format
Safetensors

Estimated size
...

Context
...

Ollama
✓ Compatible

AirLLM
✓ Compatible
```

If something cannot be determined reliably, display:

`Unknown`

Do not guess.

---

# 11. Hardware Detection

The application should detect:

* CPU
* GPU
* GPU vendor
* VRAM
* system RAM
* available disk space
* operating system

Use this information to recommend a runtime and estimate whether a model/configuration is practical.

Do not pretend an inference configuration will work simply because the model can theoretically load.

---

# 12. Error Handling

Errors must be understandable to normal users.

Bad:

```text
RuntimeError: CUDA out of memory...
```

Better:

```text
This model could not be loaded because the available GPU memory is insufficient.

Try:
• switching to AirLLM
• using a smaller/quantized model
• reducing the number of simultaneous agents
```

But retain a technical details section for debugging.

---

# 13. UI Philosophy

The application should NOT look like generic AI-generated software.

Avoid:

* excessive gradients
* unnecessary glowing cards
* fake AI branding
* excessive rounded rectangles
* huge hero sections
* meaningless dashboard statistics
* decorative animations
* generic "AI-powered" marketing language

Prioritize:

* clean
* technical
* functional
* fast
* information-dense
* professional
* desktop-native feeling

The application is an engineering/research tool.

It should feel closer to a serious developer tool than a marketing website.

---

# 14. Local-First

The primary goal is local execution.

The platform should not require a cloud account for basic local inference.

Model data should remain local unless the user explicitly chooses a remote provider.

Never upload a user's model or prompts without explicit permission.

---

# 15. Architecture Principles

Prioritize:

1. Runtime abstraction
2. Model compatibility detection
3. Agent orchestration
4. Resource management
5. Reproducible benchmarking
6. Clear error handling
7. Local-first execution
8. Extensibility

Avoid building unnecessary features before the core pipeline works.

The first milestone should be:

```text
Install
  ↓
Select Ollama/AirLLM
  ↓
Load Gemma 4 E4B
  ↓
Send prompt
  ↓
Get response
```

Second milestone:

```text
1 model
  ↓
2 independent agents
  ↓
compare responses
```

Third milestone:

```text
2 agents
  ↓
critic/debate
  ↓
final response
```

Fourth milestone:

```text
Benchmark
  ↓
measure
  ↓
compare
```

Only after these work reliably should advanced UI and additional runtimes be added.


# Critical Deployment Requirement

This application MUST NOT be designed as a traditional web application.

It must be a **standalone cross-platform desktop application**.

The target platforms are:

* Windows
* Linux
* macOS

The user should launch the application like any normal desktop application.

The user should NOT be required to:

* open a browser
* navigate to localhost
* manually start a web server
* manually run Docker commands
* manually run Python commands
* manually configure inference servers

Docker should be used as an internal deployment/runtime mechanism where appropriate.

---

# Desktop Architecture

The architecture should conceptually be:

```text
              Desktop Application
                     |
          ┌──────────┴──────────┐
          |                     |
       Native UI          Application Controller
                                |
                         Docker Manager
                                |
              ┌─────────────────┼─────────────────┐
              |                 |                 |
              v                 v                 v
           Ollama            AirLLM          Future Runtime
          Container         Container
              |                 |
              └─────────────────┼─────────────────┘
                                |
                         Runtime Interface
                                |
                         Agent Orchestrator
                                |
                         Local Model Sessions
```

The desktop application is the primary user interface.

Do not make the primary interface a browser-based frontend.

Do not require the user to access `localhost`.

---

# Docker

Docker should encapsulate runtime dependencies where practical.

The application should manage the lifecycle of its required containers.

The user should not need to understand Docker.

The application should be able to:

* detect whether Docker is installed/running
* report when Docker is unavailable
* start required containers
* stop containers when appropriate
* inspect container health
* pass model directories into containers
* persist model data outside ephemeral containers
* collect runtime logs
* recover from crashed containers

Do not store large model files inside ephemeral container layers unnecessarily.

Use appropriate host-mounted persistent directories or volumes.

---

# Cross-Platform Hardware

Do NOT assume GPU access works identically on every operating system.

Hardware detection must be separated from the inference runtime.

The application should detect, where possible:

* CPU
* GPU vendor
* GPU model
* VRAM
* system RAM
* OS
* architecture
* available storage

The runtime selection system must account for platform-specific GPU support.

For example:

```text
Windows + NVIDIA
    → CUDA-capable runtime

Linux + NVIDIA
    → CUDA-capable runtime

macOS + Apple Silicon
    → Metal/native-compatible execution where supported

CPU-only
    → CPU-compatible runtime
```

Do not pretend that a CUDA Docker container automatically provides GPU acceleration on macOS.

---

# Native Desktop Controller

The desktop application should own:

* application lifecycle
* settings
* model selection
* runtime selection
* Docker lifecycle
* hardware detection
* agent configuration
* benchmark configuration
* logs
* error reporting

The inference containers should primarily own inference.

This separation is important.

---

# No Localhost Dependency

The application may internally communicate with services through local sockets, ports, IPC, or other mechanisms.

That is an implementation detail.

The user experience MUST NOT depend on manually opening a localhost URL.

The application should communicate with internal services automatically.

Do not design the product around:

```text
Browser → localhost → backend
```

Design it around:

```text
Desktop App → Runtime Controller → Local Runtime
```

---

# Installation Experience

The desired experience is:

```text
Install application
       ↓
Launch application
       ↓
System check
       ↓
Detect Docker
       ↓
Detect hardware
       ↓
Select runtime
       ↓
Configure runtime automatically
       ↓
Import model
       ↓
Ready
```

If Docker is missing, provide a clear installation requirement/instruction rather than silently failing.

Do not make Docker setup part of the normal model-running workflow.

---

# Important Portability Principle

Do not assume that "Dockerized" automatically means "portable."

Docker provides reproducibility for the userspace/runtime environment, but GPU acceleration and hardware integration remain platform-specific.

Therefore:

```text
Portable Application
        ≠
Identical Runtime Implementation Everywhere
```

The abstraction layer must hide those differences from the rest of the application.

---

# Runtime Architecture

Keep the runtime interface platform-independent:

```text
Runtime
├── load_model()
├── unload_model()
├── generate()
├── stream()
├── get_model_info()
├── get_capabilities()
├── get_status()
└── health_check()
```

Implement platform/runtime-specific behavior behind this interface.

The agent orchestration system must never contain Docker-specific, Ollama-specific, CUDA-specific, or AirLLM-specific logic.

---

# Primary Product Principle

The user should think:

> "I installed a desktop AI application."

They should NOT think:

> "I installed a Python application that starts a localhost server which talks to Docker which runs an inference server."

All infrastructure complexity should remain behind the application.

The product should feel like a normal native desktop application while using containers internally to make model execution and dependencies reproducible.
