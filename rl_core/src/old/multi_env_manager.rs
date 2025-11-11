// Multi-environment manager for parallel Unity instances
// Each Unity instance connects to a separate gRPC server on base_port + worker_id

use crate::old::grpc_server::{GrpcServer, InitConfig};
use crate::communicator_objects::{UnityRlInitializationOutputProto, UnityOutputProto};

pub struct MultiEnvManager {
    servers: Vec<GrpcServer>,
    base_port: u16,
}

impl MultiEnvManager {
    pub fn new(base_port: u16, num_envs: usize, verbose: bool) -> Self {
        let mut servers = Vec::with_capacity(num_envs);
        
        for i in 0..num_envs {
            let port = base_port + i as u16;
            let server = GrpcServer::new_with_verbosity(port, verbose);
            servers.push(server);
        }
        
        Self {
            servers,
            base_port,
        }
    }
    
    pub fn num_envs(&self) -> usize {
        self.servers.len()
    }
    
    pub fn base_port(&self) -> u16 {
        self.base_port
    }
    
    /// Initialize all environments and wait for connections
    pub async fn initialize_all(&mut self, init_config: InitConfig, timeout_secs: u64) -> Result<Vec<UnityRlInitializationOutputProto>, String> {
        let mut specs = Vec::with_capacity(self.servers.len());
        
        for (i, server) in self.servers.iter_mut().enumerate() {
            let port = self.base_port + i as u16;
            
            let result = tokio::time::timeout(
                tokio::time::Duration::from_secs(timeout_secs),
                server.initialize(init_config.clone())
            ).await;
            
            match result {
                Ok(Ok(output)) => {
                    if let Some(init) = output.rl_initialization_output {
                        specs.push(init);
                        if !server.verbose {
                            println!("  ✓ Ambiente {} conectado na porta {}", i + 1, port);
                        }
                    } else {
                        return Err(format!("Ambiente {} (porta {}) não retornou specs", i + 1, port));
                    }
                },
                Ok(Err(e)) => {
                    return Err(format!("Ambiente {} (porta {}): {}", i + 1, port, e));
                },
                Err(_) => {
                    return Err(format!("Ambiente {} (porta {}) timeout após {}s", i + 1, port, timeout_secs));
                }
            }
        }
        
        Ok(specs)
    }
    
    /// Get mutable reference to a specific environment server
    pub fn get_server(&mut self, index: usize) -> Option<&mut GrpcServer> {
        self.servers.get_mut(index)
    }
    
    /// Step all environments in parallel
    pub async fn step_all(&mut self, actions: Vec<crate::communicator_objects::UnityInputProto>) -> Result<Vec<UnityOutputProto>, String> {
        if actions.len() != self.servers.len() {
            return Err(format!("Actions length ({}) != num_envs ({})", actions.len(), self.servers.len()));
        }
        
        let verbose = self.servers.first().map(|s| s.verbose).unwrap_or(false);
        let base_port = self.base_port;
        let mut handles = Vec::new();
        
        for (i, (server, action)) in self.servers.iter_mut().zip(actions.into_iter()).enumerate() {
            if verbose {
                println!("[MultiEnvManager] Sending action to Env {} (port {})", i + 1, base_port + i as u16);
            }
            let fut = server.step(action);
            handles.push(fut);
        }
        
        let mut outputs = Vec::new();
        for (i, fut) in handles.into_iter().enumerate() {
            let output = fut.await?;
            if verbose {
                println!("[MultiEnvManager] Received response from Env {} (port {})", i + 1, base_port + i as u16);
            }
            outputs.push(output);
        }
        
        Ok(outputs)
    }
    
    /// Reset all environments in parallel
    pub async fn reset_all(&mut self) -> Result<Vec<UnityOutputProto>, String> {
        self.reset_all_with_side_channel(vec![]).await
    }
    
    /// Reset all environments with side channel data
    pub async fn reset_all_with_side_channel(&mut self, side_channel: Vec<u8>) -> Result<Vec<UnityOutputProto>, String> {
        let mut handles = Vec::new();
        
        for server in self.servers.iter_mut() {
            let fut = server.reset_with_side_channel(side_channel.clone());
            handles.push(fut);
        }
        
        let mut outputs = Vec::new();
        for fut in handles {
            outputs.push(fut.await?);
        }
        
        Ok(outputs)
    }
}
