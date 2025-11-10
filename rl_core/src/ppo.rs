use crate::networks::{Actor, Critic};
use crate::settings::RootConfig;
use crate::communicator_objects::UnityMessageProto; // Adicionado
// crate::env_manager::UnityEnvManager não é mais usado diretamente aqui
use burn::tensor::{backend::Backend, Tensor};
use serde::{Deserialize, Serialize};
// use serde_yaml; // Not needed currently
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::collections::HashMap; // Adicionado

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PPOTrainerConfig {
    pub input_size: usize, // Deve vir do handshake
    pub action_size: usize, // Deve vir do handshake
    pub hidden_units: usize,
    pub num_layers: usize,
    pub max_steps: u64,
    pub checkpoint_interval: u64,
    pub num_envs: usize,
    pub gamma: f32,
    pub gae_lambda: f32,
    pub export_onnx_every_checkpoint: bool,
}

// Usamos os tipos de grpc.rs
use crate::communicator_objects::unity_to_external_proto_client;

pub struct PPOTrainer<B: Backend> {
    actor: Actor<B>,
    critic: Critic<B>,
    device: B::Device,
    cfg: PPOTrainerConfig,
    // Agora clients são criados e mantidos aqui, após inicialização feita externamente (em run_from_config)
    clients: Vec<unity_to_external_proto_client::UnityToExternalProtoClient<tonic::transport::Channel>>,
    addresses: Vec<std::net::SocketAddr>, // Para referência
}

impl<B: Backend> PPOTrainer<B> {
    // Agora new assume que a inicialização (e descoberta de specs) já foi feita externamente.
    // Recebe cfg com tamanhos corretos e addresses para conectar e montar a lista de clients.
    pub fn new(device: &B::Device, cfg: PPOTrainerConfig, addresses: Vec<std::net::SocketAddr>) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let actor = Actor::new(
            cfg.input_size,
            cfg.hidden_units,
            cfg.num_layers,
            cfg.action_size,
            device,
        );
        let critic = Critic::new(cfg.input_size, cfg.hidden_units, cfg.num_layers, device);

        // Conectar aos Unitys baseado nos addresses, assumindo conexão inicial já feita
        let rt = tokio::runtime::Runtime::new()?;
        let clients: Vec<_> = rt.block_on(async {
            let mut client_list = Vec::new();
            for addr in &addresses {
                let client = crate::grpc::connect_to_unity(*addr).await?;
                client_list.push(client);
            }
            Ok::<Vec<_>, Box<dyn std::error::Error + Send + Sync>>(client_list)
        })?;

        Ok(Self {
            actor,
            critic,
            device: device.clone(),
            cfg,
            clients, // Lista de clients prontos para trocar mensagens de step
            addresses,
        })
    }

    pub fn train(&mut self) {
        // Novo loop: troca de mensagens com Unity para steps
        // Inicializar ações iniciais (ex: zero)
        let initial_actions: Vec<Vec<f32>> = vec![vec![0.0; self.cfg.action_size]; self.cfg.num_envs];
        let _current_actions = initial_actions; // Se não for usada, adicione _; se for usada, remova o mut

        let mut buf = crate::ppo_buffer::RolloutBuffer::new();
        // Estado para cada ambiente
        let mut current_obs_per_env = vec![Vec::new(); self.cfg.num_envs];
        let mut current_reward_per_env = vec![0.0f32; self.cfg.num_envs];
        let mut current_done_per_env = vec![false; self.cfg.num_envs];

        let rt = tokio::runtime::Runtime::new().unwrap(); // Runtime para as chamadas assíncronas

        for step in 1..=self.cfg.max_steps {
            // Calcular todas as ações antes das trocas
            let next_actions: Vec<Vec<f32>> = (0..self.cfg.num_envs).map(|i| {
                // Obter ação do modelo com base na observação atual (ou ação inicial)
                let obs = &current_obs_per_env[i];
                let obs_len = obs.len().max(self.cfg.input_size.max(1));
                let mut obs_vec = obs.clone();
                if obs_vec.len() < obs_len { obs_vec.resize(obs_len, 0.0); }
                let obs_t = Tensor::<B,2>::from_floats(obs_vec.as_slice(), &self.device).reshape([1, obs_len]);

                let act_t = self.actor.forward(obs_t.clone());
                let data = act_t.into_data();
                let bytes = data.bytes;
                let floats: &[f32] = unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, bytes.len() / std::mem::size_of::<f32>()) };
                let mut action = Vec::new();
                action.extend_from_slice(&floats[0..self.cfg.action_size.min(floats.len())]);
                action
            }).collect();

            // Vetores para armazenar resultados das trocas
            let mut new_obs_per_env = vec![None; self.cfg.num_envs];
            let mut new_rewards = vec![0.0f32; self.cfg.num_envs];
            let mut new_done = vec![false; self.cfg.num_envs];
            let mut transitions_to_store = vec![];

            // Realizar as trocas de forma síncrona (mas com runtime para o gRPC)
            for (i, client) in self.clients.iter_mut().enumerate() {
                let action_to_send = next_actions[i].clone();
                // Montar mensagem de input para step
                use crate::communicator_objects::{UnityInputProto, UnityRlInputProto, unity_rl_input_proto::ListAgentActionProto, AgentActionProto, CommandProto};
                let agent_action_proto = AgentActionProto {
                    vector_actions_deprecated: vec![],
                    value: 0.0,
                    continuous_actions: action_to_send, // Enviar a ação calculada anteriormente
                    discrete_actions: vec![], // TODO: lidar com ações discretas
                };
                let mut action_map = HashMap::new();
                // Assumir um behavior fixo para este trainer
                action_map.insert("Behavior".to_string(), ListAgentActionProto { value: vec![agent_action_proto] });

                let rl_input = UnityRlInputProto {
                    agent_actions: action_map,
                    command: CommandProto::Step as i32,
                    side_channel: vec![], // TODO
                };
                let input_msg = UnityMessageProto {
                    header: None, // Preenchido pelo Unity
                    unity_output: None,
                    unity_input: Some(UnityInputProto { rl_input: Some(rl_input), rl_initialization_input: None }),
                };

                // Enviar e receber (blocking call using rt)
                let response_msg = rt.block_on(async {
                    crate::grpc::exchange_with_unity(client, input_msg).await
                }).unwrap_or_else(|e| {
                    eprintln!("[erro] Falha na troca gRPC com o ambiente {}: {}", i, e);
                    // Mensagem padrão em caso de erro
                    UnityMessageProto { header: None, unity_output: None, unity_input: None }
                });

                // Processar resposta
                if let (Some((rew, done)), Some(obs)) = (crate::grpc::parse_step_reward_done(&response_msg), crate::grpc::parse_step_observation_flat(&response_msg)) {
                    new_obs_per_env[i] = Some(obs.clone());
                    new_rewards[i] = rew;
                    new_done[i] = done;
                    // Armazenar transição para o buffer
                    transitions_to_store.push((rew, 0.0, done)); // value_placeholder é 0.0
                }
            }

            // Atualizar estados após todas as trocas
            for (i, new_obs) in new_obs_per_env.into_iter().enumerate() {
                if let Some(obs) = new_obs {
                    current_obs_per_env[i] = obs;
                    current_reward_per_env[i] = new_rewards[i];
                    current_done_per_env[i] = new_done[i];
                }
            }

            // Armazenar transições no buffer
            for (rew, val, done) in transitions_to_store {
                buf.push(vec![], vec![], rew, val, done, 0.0); // placeholder obs/actions
            }

            // Processar buffer e salvar checkpoint se necessário
            if step % self.cfg.checkpoint_interval == 0 {
                // Calcular GAE e atualizar buffer, critic network, etc. (placeholder)
                buf.finish_path(self.cfg.gamma, self.cfg.gae_lambda, 0.0);
                // Treinamento real do actor/critic (placeholder)
                buf.clear();
                let _ = self.save_checkpoint(step);
                if self.cfg.export_onnx_every_checkpoint { let _ = self.export_onnx(step); }
            }
        }
    }

    fn checkpoints_dir() -> PathBuf { PathBuf::from("checkpoints") }

    fn save_checkpoint(&self, step: u64) -> std::io::Result<PathBuf> {
        let dir = Self::checkpoints_dir();
        fs::create_dir_all(&dir)?;
        let path = dir.join(format!("step_{step}.json"));
        let meta = serde_json::json!({
            "step": step,
            "input_size": self.cfg.input_size,
            "hidden_units": self.cfg.hidden_units,
            "num_layers": self.cfg.num_layers,
            "action_size": self.cfg.action_size,
        });
        fs::write(&path, serde_json::to_vec_pretty(&meta).unwrap())?;
        Ok(path)
    }

    fn export_onnx(&self, step: u64) -> std::io::Result<()> {
        let script_path = PathBuf::from("rl_core").join("export_onnx.py");
        if !script_path.exists() { return Ok(()); }
        let out_dir = Self::checkpoints_dir();
        fs::create_dir_all(&out_dir)?;
        let onnx_out = out_dir.join(format!("step_{step}.onnx"));
        let status = Command::new("python3").arg(script_path).arg("--out").arg(&onnx_out).status();
        match status {
            Ok(s) if s.success() => Ok(()),
            _ => Ok(()),
        }
    }
}

// Atualizar run_from_config para fazer inicialização e descoberta de specs
pub fn run_from_config<B: Backend>(device: &B::Device, root: &RootConfig, behavior: &str) {
    // Pick behavior
    if let Some(bcfg) = root.behaviors.get(behavior) {
        let hidden = bcfg.network_settings.as_ref().and_then(|n| n.hidden_units).unwrap_or(128);
        let layers = bcfg.network_settings.as_ref().and_then(|n| n.num_layers).unwrap_or(2);
        let num_envs = root.env_settings.as_ref().and_then(|e| e.num_envs).unwrap_or(1);
        let max_steps = bcfg.max_steps;
        let ckpt = bcfg.checkpoint_interval as u64;

        // Configure gRPC init settings (mesmo código)
        if let Some(es) = &root.env_settings {
            let cfg = crate::grpc::InitConfig {
                seed: es.seed.unwrap_or(-1),
                num_areas: es.num_areas.unwrap_or(1),
                communication_version: "1.5.0".to_string(),
                package_version: "0.1.0".to_string(),
            };
            crate::grpc::set_init_config(cfg);
        }

        // Env processes are managed by CLI main; don't restart here
        // (Unity já iniciado pelo CLI)

        // --- NOVO: Descoberta de specs via cliente ---
        let env_params = root.env_settings.as_ref().and_then(|es| es.environment_parameters.clone()).unwrap_or_default();

        let rt_init = tokio::runtime::Runtime::new().unwrap();
        let mut specs_per_env = HashMap::new();
        let base_port = root.env_settings.as_ref().and_then(|e| e.base_port).unwrap_or(5005);
        let mut addrs = Vec::new();
        for i in 0..num_envs {
            addrs.push(std::net::SocketAddr::from(([127,0,0,1], base_port + i as u16)));
        }

        // Conectar e inicializar cada Unity com parametros
        // Esperar por conexão se necessário, com timeout
        let timeout_secs = 30; // Exemplo de timeout
        let start_time = std::time::Instant::now();
        loop {
            let mut all_connected = true;
            for addr in &addrs {
                if !specs_per_env.contains_key(addr) {
                    match rt_init.block_on(async { crate::grpc::connect_to_unity(*addr).await }) {
                        Ok(mut client) => {
                            match rt_init.block_on(async { crate::grpc::initialize_unity_with_params(&mut client, env_params.clone()).await }) {
                                Ok(specs) => { specs_per_env.insert(*addr, specs); },
                                Err(e) => { eprintln!("[debug] Inicialização com {} falhou: {}, tentando novamente...", addr, e); all_connected = false; }
                            }
                        },
                        Err(e) => { eprintln!("[debug] Conexão a {} falhou: {}, tentando novamente...", addr, e); all_connected = false; }
                    }
                }
            }
            if all_connected { break; }
            if start_time.elapsed().as_secs() > timeout_secs {
                eprintln!("[erro] Timeout esperando por inicialização do Unity em todas as portas: {:?}", addrs);
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(500)); // Espera antes de tentar novamente
        }

        if specs_per_env.is_empty() {
            eprintln!("[erro] Nenhuma conexão Unity bem-sucedida após inicialização.");
            return;
        }

        // Pegar specs do primeiro ambiente como referência (mesmo código do antigo)
        let first_specs = specs_per_env.values().next().unwrap();
        let obs_sum: usize = first_specs.observation_sizes.iter().copied().sum();
        let input_size = obs_sum.max(1);
        let action_size = first_specs.action_size.max(1);

        // Derive gamma and lambda (mesmo código)
        let gamma = bcfg.reward_signals.as_ref().and_then(|m| m.get("extrinsic")).and_then(|e| e.gamma).unwrap_or(0.99);
        let gae_lambda = bcfg.hyperparameters.lambd;

        // Build trainer com specs descobertos
        let mut trainer: PPOTrainer<B> = match PPOTrainer::new(
            device,
            PPOTrainerConfig {
                input_size,
                action_size,
                hidden_units: hidden,
                num_layers: layers,
                max_steps,
                checkpoint_interval: ckpt,
                num_envs,
                gamma,
                gae_lambda,
                export_onnx_every_checkpoint: true,
            },
            addrs,
        ) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("[erro] Falha ao criar PPOTrainer: {}", e);
                return;
            }
        };

        trainer.train(); // Agora o train usa o novo ciclo
    }
}