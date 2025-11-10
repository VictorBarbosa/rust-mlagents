use tonic::Request;
use tonic::transport::Channel;
use std::net::SocketAddr;
use std::sync::{Mutex, OnceLock, Arc};
use std::sync::atomic::{AtomicBool, Ordering};

use crate::communicator_objects::UnityMessageProto;
use crate::communicator_objects::unity_to_external_proto_client::UnityToExternalProtoClient;

#[derive(Debug, Clone, Default)]
pub struct Specs {
    pub observation_sizes: Vec<usize>,
    pub action_size: usize,
}

static LATEST_SPECS: OnceLock<Mutex<Option<Specs>>> = OnceLock::new();

// Global policy hook: maps flattened observations for a single agent -> continuous actions
static ACTION_PROVIDER: OnceLock<Arc<dyn Fn(&[f32], usize) -> Vec<f32> + Send + Sync>> = OnceLock::new();
// Trainer-provided next action vector (used if set)
static NEXT_ACTION: OnceLock<Mutex<Vec<f32>>> = OnceLock::new();

#[derive(Clone, Debug, Default)]
pub struct StepData {
    pub obs: Vec<f32>,
    pub reward: f32,
    pub done: bool,
}

static LATEST_STEP: OnceLock<Mutex<Option<StepData>>> = OnceLock::new();

static CONNECTED: OnceLock<AtomicBool> = OnceLock::new();
fn connected_cell() -> &'static AtomicBool { CONNECTED.get_or_init(|| AtomicBool::new(false)) }
pub fn mark_connected() { connected_cell().store(true, Ordering::SeqCst); }
pub fn is_connected() -> bool { connected_cell().load(Ordering::SeqCst) }

fn specs_cell() -> &'static Mutex<Option<Specs>> { LATEST_SPECS.get_or_init(|| Mutex::new(None)) }
fn next_action_cell() -> &'static Mutex<Vec<f32>> { NEXT_ACTION.get_or_init(|| Mutex::new(Vec::new())) }
fn step_cell() -> &'static Mutex<Option<StepData>> { LATEST_STEP.get_or_init(|| Mutex::new(None)) }

pub fn get_latest_specs() -> Option<Specs> {
    match specs_cell().lock() {
        Ok(guard) => (*guard).clone(),
        Err(_) => None,
    }
}

pub fn set_action_provider(p: Arc<dyn Fn(&[f32], usize) -> Vec<f32> + Send + Sync>) { let _ = ACTION_PROVIDER.set(p); }
fn get_action_provider() -> Option<&'static Arc<dyn Fn(&[f32], usize) -> Vec<f32> + Send + Sync>> { ACTION_PROVIDER.get() }

pub fn set_next_action(a: Vec<f32>) { if let Ok(mut g) = next_action_cell().lock() { *g = a; } }
pub fn set_latest_step(data: StepData) { if let Ok(mut g) = step_cell().lock() { *g = Some(data); } }
pub fn get_latest_step() -> Option<StepData> { step_cell().lock().ok().and_then(|g| (*g).clone()) }

// Updates observation_sizes in LATEST_SPECS if they are empty
fn update_observation_sizes_if_empty(obs_len: usize) {
    if let Ok(mut specs_guard) = specs_cell().lock() {
        if let Some(ref mut specs) = *specs_guard {
            if specs.observation_sizes.is_empty() && obs_len > 0 {
                // For now, assume a single observation space of size obs_len
                // In a more complex scenario with multiple named observations, this would be a vector of sizes
                specs.observation_sizes = vec![obs_len];
            }
        }
    }
}

fn try_specs_from_initialization_message(msg: &UnityMessageProto) -> Option<Specs> {
    // Attempt to infer observation and action specs from initialization output
    if let Some(out) = &msg.unity_output {
        if let Some(init) = &out.rl_initialization_output {
            // Use first brain parameters for now (multi-brain is a future extension)
            if let Some(bp) = init.brain_parameters.first() {

                // Observation sizes are not directly in BrainParametersProto in v1.5.0
                // They are determined by the agent's first observation in a subsequent step.
                // However, for initialization handshake, we might not have that yet.
                // In the Python mlagents, the specs are fully known after initialization.
                // For now, we just extract action specifications and leave obs_sizes empty.
                // The actual obs_sizes will be inferred on the first step data received.
                // This function focuses on the static part of the spec (action space).
                if let Some(spec) = &bp.action_spec {
                    let cont = spec.num_continuous_actions.max(0) as usize;
                    let disc_sum: usize = spec.discrete_branch_sizes.iter().map(|v| (*v).max(0) as usize).sum();
                    let action_size = cont + disc_sum;
                    // We return Specs with known action size and empty obs_sizes.
                    // The obs_sizes must be inferred later from actual step data.
                    return Some(Specs { observation_sizes: Vec::new(), action_size });
                }
            }
        }
    }
    None
}

pub fn parse_step_reward_done(msg: &UnityMessageProto) -> Option<(f32, bool)> {
    if let Some(out) = &msg.unity_output {
        if let Some(rl_out) = &out.rl_output {
            // Pick first behavior, first agent
            for (_beh, list) in rl_out.agent_infos.iter() {
                if let Some(agent) = list.value.first() {
                    return Some((agent.reward, agent.done));
                }
            }
        }
    }
    None
}

pub fn parse_step_observation_flat(msg: &UnityMessageProto) -> Option<Vec<f32>> {
    if let Some(out) = &msg.unity_output {
        if let Some(rl_out) = &out.rl_output {
            for (_beh, list) in rl_out.agent_infos.iter() {
                if let Some(agent) = list.value.first() {
                    // Concatenate all observation float_data for the agent
                    let mut data = Vec::new();
                    for obs in agent.observations.iter() {
                        if let Some(obs_data) = &obs.observation_data {
                            match obs_data {
                                crate::communicator_objects::observation_proto::ObservationData::FloatData(fd) => {
                                    data.extend_from_slice(&fd.data);
                                }
                                crate::communicator_objects::observation_proto::ObservationData::CompressedData(_) => {}
                            }
                        }
                    }
                    if !data.is_empty() { return Some(data); }
                }
            }
        }
    }
    None
}

#[derive(Clone, Debug)]
pub struct InitConfig {
    pub seed: i32,
    pub num_areas: i32,
    pub communication_version: String,
    pub package_version: String,
}

impl Default for InitConfig {
    fn default() -> Self {
        Self { seed: -1, num_areas: 1, communication_version: "1.5.0".to_string(), package_version: "0.1.0".to_string() }
    }
}

static INIT_CFG: OnceLock<Mutex<InitConfig>> = OnceLock::new();

fn init_cfg_cell() -> &'static Mutex<InitConfig> {
    INIT_CFG.get_or_init(|| Mutex::new(InitConfig::default()))
}

pub fn set_init_config(cfg: InitConfig) {
    if let Ok(mut g) = init_cfg_cell().lock() { *g = cfg; }
}

pub fn get_init_config() -> InitConfig {
    init_cfg_cell().lock().ok().map(|g| g.clone()).unwrap_or_default()
}

// --- Novas funções para cliente gRPC ---

pub async fn connect_to_unity(addr: SocketAddr) -> Result<UnityToExternalProtoClient<Channel>, Box<dyn std::error::Error + Send + Sync>> {
    let endpoint = format!("http://{}", addr);
    let client = UnityToExternalProtoClient::connect(endpoint).await?;
    Ok(client)
}

// Função para enviar a mensagem de inicialização e receber a resposta
pub async fn initialize_unity_connection(client: &mut UnityToExternalProtoClient<Channel>) -> Result<Specs, Box<dyn std::error::Error + Send + Sync>> {
    use crate::communicator_objects::{UnityInputProto, UnityRlInitializationInputProto, UnityRlCapabilitiesProto};

    let cfg = get_init_config();
    let caps = UnityRlCapabilitiesProto {
        base_rl_capabilities: true,
        concatenated_png_observations: true,
        compressed_channel_mapping: true,
        hybrid_actions: true,
        training_analytics: false,
        variable_length_observation: true,
        multi_agent_groups: true,
    };
    let init_in = UnityRlInitializationInputProto {
        seed: cfg.seed,
        communication_version: cfg.communication_version,
        package_version: cfg.package_version,
        capabilities: Some(caps),
        num_areas: cfg.num_areas,
        environment_parameters: std::collections::HashMap::new(), // Empty for standalone test
    };
    let init_msg = UnityMessageProto {
        header: None, // O header é populado pelo Unity
        unity_output: None,
        unity_input: Some(UnityInputProto { rl_input: None, rl_initialization_input: Some(init_in) }),
    };

    let request = Request::new(init_msg);
    let response = client.exchange(request).await?;
    let response_msg = response.into_inner();

    // Processar a resposta de inicialização do Unity
    if let Some(output) = &response_msg.unity_output {
        if let Some(_init_output) = &output.rl_initialization_output {
            // Extrair specs da resposta
            if let Some(specs) = try_specs_from_initialization_message(&response_msg) {
                // Salvar specs recebidos
                if let Ok(mut specs_guard) = specs_cell().lock() {
                    *specs_guard = Some(specs.clone());
                }
                mark_connected();
                return Ok(specs);
            } else {
                return Err("Falha ao parsear specs da resposta de inicialização do Unity".into());
            }
        }
    }

    Err("Resposta de inicialização do Unity não contém dados esperados".into())
}


// Função genérica para trocar uma mensagem com o Unity (útil para steps)
pub async fn exchange_with_unity(
    client: &mut UnityToExternalProtoClient<Channel>,
    msg: UnityMessageProto,
) -> Result<UnityMessageProto, Box<dyn std::error::Error + Send + Sync>> {
    let request = Request::new(msg);
    let response = client.exchange(request).await?;
    Ok(response.into_inner())
}

// --- Funções para lidar com environment_parameters ---

pub async fn initialize_unity_with_params(
    client: &mut UnityToExternalProtoClient<Channel>,
    env_params: std::collections::HashMap<String, serde_yaml::Value>,
) -> Result<Specs, Box<dyn std::error::Error + Send + Sync>> {
    use crate::communicator_objects::{UnityInputProto, UnityRlInitializationInputProto, UnityRlCapabilitiesProto};

    let cfg = get_init_config();
    let caps = UnityRlCapabilitiesProto {
        base_rl_capabilities: true,
        concatenated_png_observations: true,
        compressed_channel_mapping: true,
        hybrid_actions: true,
        training_analytics: false,
        variable_length_observation: true,
        multi_agent_groups: true,
    };
    let init_in = UnityRlInitializationInputProto {
        seed: cfg.seed,
        communication_version: cfg.communication_version,
        package_version: cfg.package_version,
        capabilities: Some(caps),
        num_areas: cfg.num_areas,
        environment_parameters: std::collections::HashMap::new(), // Empty for standalone test
    };

    // --- MONTAR side_channel_data para EnvironmentParametersChannel ---
    let channel_id_bytes = [
        0xa9, 0xe4, 0xb8, 0x8d,
        0x92, 0xe9,
        0x41, 0x36,
        0x93, 0xb3,
        0x12, 0xa7, 0xc2, 0x34, 0xf4, 0xc0
    ];

    let mut side_channel_data = Vec::new();
    side_channel_data.extend_from_slice(&channel_id_bytes);

    let num_params = env_params.len() as i32;
    side_channel_data.extend_from_slice(&num_params.to_le_bytes());

    for (key, value) in env_params {
        let key_len = key.len() as i32;
        side_channel_data.extend_from_slice(&key_len.to_le_bytes());
        side_channel_data.extend_from_slice(key.as_bytes());

        let float_val = match value {
            serde_yaml::Value::Number(n) => n.as_f64().unwrap_or(0.0) as f32,
            serde_yaml::Value::String(s) => s.parse::<f32>().unwrap_or(0.0),
            _ => 0.0,
        };
        side_channel_data.extend_from_slice(&float_val.to_le_bytes());
    }
    // ---

    // Montar mensagem de inicialização COM side_channel
    // O side_channel é um campo de UnityRlInputProto.
    // A mensagem inicial pode conter tanto rl_initialization_input quanto rl_input (com side_channel).
    use std::collections::HashMap;

    let rl_input_with_sidechannel = crate::communicator_objects::UnityRlInputProto {
        agent_actions: HashMap::new(), // Vazio, já que é para env params
        command: crate::communicator_objects::CommandProto::Step as i32, // Pode ser Step
        side_channel: side_channel_data, // <-- DADOS DOS PARAMS AQUI
    };

    let init_msg = UnityMessageProto {
        header: None, // Preenchido pelo Unity
        unity_output: None,
        unity_input: Some(UnityInputProto {
            rl_input: Some(rl_input_with_sidechannel), // <-- INCLUÍDO O PROTOBUF DE RL_INPUT AQUI
            rl_initialization_input: Some(init_in), // Mensagem de inicialização original
        }),
    };

    let request = Request::new(init_msg);
    let response = client.exchange(request).await?;
    let response_msg = response.into_inner();

    // Processar a resposta de inicialização do Unity
     if let Some(output) = &response_msg.unity_output {
        if let Some(_init_output) = &output.rl_initialization_output {
            if let Some(specs) = try_specs_from_initialization_message(&response_msg) {
                if let Ok(mut specs_guard) = specs_cell().lock() {
                    *specs_guard = Some(specs.clone());
                }
                mark_connected();
                return Ok(specs);
            } else {
                return Err("Falha ao parsear specs da resposta de inicialização do Unity".into());
            }
        }
    }

    Err("Resposta de inicialização do Unity não contém dados esperados".into())
}