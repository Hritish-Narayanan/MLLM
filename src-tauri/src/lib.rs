use serde_json::Value;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use tauri::State;

struct ActiveBackend {
    _child: Child,
    stdin: ChildStdin,
    reader: BufReader<ChildStdout>,
}

pub struct BackendState {
    backend: Mutex<Option<ActiveBackend>>,
    counter: AtomicU64,
}

impl BackendState {
    pub fn new() -> Self {
        Self {
            backend: Mutex::new(None),
            counter: AtomicU64::new(1),
        }
    }

    pub fn ensure_started(&self) -> Result<(), String> {
        let mut guard = self.backend.lock().map_err(|e| e.to_string())?;
        if guard.is_none() {
            let current_exe_dir = std::env::current_exe()
                .ok()
                .and_then(|p| p.parent().map(|d| d.to_path_buf()));

            let host_target = if cfg!(target_os = "windows") {
                "x86_64-pc-windows-msvc"
            } else if cfg!(target_os = "linux") {
                "x86_64-unknown-linux-gnu"
            } else if cfg!(target_arch = "aarch64") {
                "aarch64-apple-darwin"
            } else {
                "x86_64-apple-darwin"
            };

            let binary_ext = if cfg!(target_os = "windows") { ".exe" } else { "" };
            let sidecar_target = format!("mllm-backend-{}{}", host_target, binary_ext);
            let sidecar_simple = format!("mllm-backend{}", binary_ext);

            // Comprehensive candidate locations
            let mut candidates = Vec::new();
            if let Some(ref d) = current_exe_dir {
                candidates.push(d.join(&sidecar_simple));
                candidates.push(d.join(&sidecar_target));
                candidates.push(d.join("binaries").join(&sidecar_simple));
                candidates.push(d.join("binaries").join(&sidecar_target));
                candidates.push(d.join("../Resources/binaries").join(&sidecar_simple));
                candidates.push(d.join("../Resources/binaries").join(&sidecar_target));
            }
            if let Ok(cur) = std::env::current_dir() {
                candidates.push(cur.join("src-tauri/binaries").join(&sidecar_simple));
                candidates.push(cur.join("src-tauri/binaries").join(&sidecar_target));
                candidates.push(cur.join("binaries").join(&sidecar_simple));
                candidates.push(cur.join("binaries").join(&sidecar_target));
            }

            let mut sidecar_cmd = None;
            for candidate in candidates {
                if candidate.exists() {
                    sidecar_cmd = Some(Command::new(candidate));
                    break;
                }
            }

            let mut child = if let Some(mut cmd) = sidecar_cmd {
                cmd.stdin(Stdio::piped())
                    .stdout(Stdio::piped())
                    .stderr(Stdio::inherit())
                    .spawn()
                    .map_err(|e| format!("Failed to spawn bundled sidecar binary: {}", e))?
            } else {
                // Fallback to local python venv
                let py_candidates = vec![
                    std::env::current_dir().ok().map(|p| p.join(".venv/bin/python")),
                    std::env::current_dir().ok().map(|p| p.join(".venv/Scripts/python.exe")),
                ];

                let mut py_path = "python3".to_string();
                for cand in py_candidates.into_iter().flatten() {
                    if cand.exists() {
                        py_path = cand.to_string_lossy().to_string();
                        break;
                    }
                }

                Command::new(&py_path)
                    .arg("backend/ipc/server.py")
                    .stdin(Stdio::piped())
                    .stdout(Stdio::piped())
                    .stderr(Stdio::inherit())
                    .spawn()
                    .map_err(|e| format!("Failed to spawn Python backend: {}", e))?
            };

            let stdin = child.stdin.take().ok_or("Failed to open child stdin")?;
            let stdout = child.stdout.take().ok_or("Failed to open child stdout")?;
            let reader = BufReader::new(stdout);

            *guard = Some(ActiveBackend {
                _child: child,
                stdin,
                reader,
            });
        }
        Ok(())
    }

    pub fn call_rpc(&self, method: &str, params: Value) -> Result<Value, String> {
        self.ensure_started()?;

        let mut guard = self.backend.lock().map_err(|e| e.to_string())?;
        let active = guard.as_mut().ok_or("Backend process not running")?;

        let id = self.counter.fetch_add(1, Ordering::SeqCst);
        let req = serde_json::json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        });

        let req_str = req.to_string();
        writeln!(active.stdin, "{}", req_str)
            .map_err(|e| format!("Failed to write to stdin: {}", e))?;
        active.stdin.flush().map_err(|e| format!("Failed to flush stdin: {}", e))?;

        let mut line = String::new();
        active.reader
            .read_line(&mut line)
            .map_err(|e| format!("Failed to read line from stdout: {}", e))?;

        if line.trim().is_empty() {
            return Err("Empty response from backend".to_string());
        }

        let resp: Value = serde_json::from_str(&line)
            .map_err(|e| format!("Failed to parse JSON-RPC response '{}': {}", line, e))?;

        if let Some(err) = resp.get("error") {
            let msg = err.get("message").and_then(|m| m.as_str()).unwrap_or("RPC error");
            return Err(msg.to_string());
        }

        Ok(resp.get("result").cloned().unwrap_or(Value::Null))
    }
}

#[tauri::command]
fn rpc(method: String, params: Option<Value>, state: State<'_, BackendState>) -> Result<Value, String> {
    state.call_rpc(&method, params.unwrap_or(Value::Null))
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let backend_state = BackendState::new();

    tauri::Builder::default()
        .manage(backend_state)
        .plugin(
            tauri_plugin_log::Builder::default()
                .level(log::LevelFilter::Info)
                .build(),
        )
        .invoke_handler(tauri::generate_handler![rpc])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
