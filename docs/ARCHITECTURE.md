# ARCHITECTURE

## Overview

Image Studio is split into two main parts:

- Python backend
- C# WPF desktop client

The backend is responsible for inference, model management, analysis, queueing, and runtime diagnostics.

The client is responsible for user workflows, previews, polling, history, presets, runtime display, and future UI theming/settings.

---

## High-Level Architecture

```text
+----------------------+
|  WPF Desktop Client  |
|----------------------|
| Server Profiles      |
| Prompt Editing       |
| Image Selection      |
| Presets              |
| History              |
| Runtime Info UI      |
| Job Polling          |
| Cancel Job           |
+----------+-----------+
           |
           | HTTP / JSON / Multipart
           v
+----------------------+
|   FastAPI Backend    |
|----------------------|
| Routes               |
| Job Queue            |
| Generation Service   |
| Analysis Services    |
| Model Registry       |
| Image Serving        |
| Runtime Diagnostics  |
+----------+-----------+
           |
           v
+----------------------+
| Local Storage        |
|----------------------|
| data/models          |
| data/inputs          |
| data/outputs         |
| data/models.json     |
+----------------------+