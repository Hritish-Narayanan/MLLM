use serde_json::Value;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use tauri::State;

pub struct BackendState {
    pub process: Mutex<Option<Child>>,
    pub counter: AtomicU64,
}

impl BackendState {
    pub fn new() -> Self {
        Self {
            process: Mutex::new(None),
            counter: AtomicU64::new(1),
        }
    }

    pub fn ensure_started(&self) -> Result<(), String> {
        let mut proc_guard = self.process.lock().map_err(|e| e.to_string())?;
        if proc_guard.is_none() {
            // 1. Check if bundled standalone sidecar binary exists alongside app or in src-tauri/binaries
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
            let sidecar_name = format!("mllm-backend-{}{}", host_target, binary_ext);

            let candidate_sidecars = vec![
                current_exe_dir.as_ref().map(|d| d.join(&sidecar_name)),
                current_exe_dir.as_ref().map(|d| d.join("binaries").join(&sidecar_name)),
                current_exe_dir.as_ref().map(|d| d.join("../Resources/binaries").join(&sidecar_name)),
                std::env::current_dir().ok().map(|d| d.join("src-tauri/binaries").join(&sidecar_name)),
                std::env::current_dir().ok().map(|d| d.join("binaries").join(&sidecar_name)),
            ];

            let mut sidecar_cmd = None;
            for candidate in candidate_sidecars.into_iter().flatten() {
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
                    .map_err(|e| format!("Failed to spawn bundled sidecar: {}", e))?
            } else {
                // Fallback to local python venv
                let py_path = std::env::current_dir()
                    .map(|p| p.join(".venv/bin/python"))
                    .unwrap_or_else(|_| std::path::PathBuf::from(".venv/bin/python"));

                let python_bin = if py_path.exists() {
                    py_path.to_string_lossy().to_string()
                } else {
                    "python3".to_string()
                };

                Command::new(&python_bin)
                    .arg("backend/ipc/server.py")
                    .stdin(Stdio::piped())
                    .stdout(Stdio::piped())
                    .stderr(Stdio::inherit())
                    .spawn()
                    .map_err(|e| format!("Failed to spawn Python backend: {}", e))?
            };

            *proc_guard = Some(child);
        }
        Ok(())
    }

    pub fn call_rpc(&self, method: &str, params: Value) -> Result<Value, String> {
        self.ensure_started()?;

        let mut proc_guard = self.process.lock().map_err(|e| e.to_string())?;
        let child = proc_guard.as_mut().ok_or("Backend process not running")?;

        let id = self.counter.fetch_add(1, Ordering::SeqCst);
        let req = serde_json::json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        });

        let mut stdin = child.stdin.as_mut().ok_or("Cannot get child stdin")?;
        let req_str = req.to_string();
        writeln!(stdin, "{}", req_str).map_err(|e| format!("Failed to write to stdin: {}", e))?;
        stdin.flush().map_err(|e| format!("Failed to flush stdin: {}", e))?;

        let stdout = child.stdout.as_mut().ok_or("Cannot get child stdout")?;
        let mut reader = BufReader::new(stdout);
        let mut line = String::new();
        reader
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
