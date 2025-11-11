use crate::old::settings::EnvSettings;
use std::process::{Child, Command, Stdio};
use std::path::PathBuf;
use std::ffi::OsStr;

#[derive(Debug, Clone)]
pub struct UnityEnvManager {
    pub env_path: Option<String>,
    pub base_port: u16,
    pub num_envs: usize,
}

impl UnityEnvManager {
    pub fn from_settings(s: &Option<EnvSettings>) -> Self {
        let (env_path, base_port, num_envs) = match s {
            Some(es) => (
                es.env_path.clone(),
                es.base_port.unwrap_or(5005),
                es.num_envs.unwrap_or(1),
            ),
            None => (None, 5005, 1),
        };
        Self { env_path, base_port, num_envs }
    }

    pub fn start_all(&self) -> std::io::Result<Vec<Child>> {
        let mut children = Vec::with_capacity(self.num_envs);
        if let Some(path_str) = &self.env_path {
            let mut app_path = PathBuf::from(path_str);
            if !app_path.exists() {
                // Try macOS .app bundle suffix if missing
                let try_app = PathBuf::from(format!("{}.app", path_str));
                if try_app.exists() { app_path = try_app; }
            }
            // If it's a macOS .app bundle, resolve inner executable instead of using `open -W` (which blocks Unity init)
            let mut inner_exec: Option<PathBuf> = None;
            if cfg!(target_os = "macos") && app_path.extension() == Some(OsStr::new("app")) {
                let contents = app_path.join("Contents").join("MacOS");
                if contents.exists() {
                    // Pick first file in Contents/MacOS as executable
                    if let Ok(mut rd) = std::fs::read_dir(&contents) {
                        if let Some(Ok(entry)) = rd.next() { inner_exec = Some(entry.path()); }
                    }
                }
            }
            for i in 0..self.num_envs {
                let port = self.base_port + i as u16;
                let worker_id = i.to_string();
                
                // Build command with worker-id for process isolation
                let mut cmd = if let Some(exec_path) = inner_exec.as_ref() {
                    Command::new(exec_path)
                } else {
                    Command::new(&app_path)
                };
                
                cmd.arg("--mlagents-port")
                   .arg(port.to_string())
                   .arg("--worker-id")
                   .arg(worker_id)
                   .stdout(Stdio::null())
                   .stderr(Stdio::null());
                
                // Add a small delay between launches to avoid port conflicts
                if i > 0 {
                    std::thread::sleep(std::time::Duration::from_millis(500));
                }
                
                let child = cmd.spawn()?;
                children.push(child);
            }
        }
        Ok(children)
    }
}
