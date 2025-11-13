# 🔍 Detecção Automática de RayPerceptionSensor

## ✨ Sistema Inteligente

O treinamento agora **detecta automaticamente** se o Unity Agent está usando RayPerceptionSensor!

## 🎯 Como Funciona

### 1. Detecção na Inicialização

Quando o treinamento começa, o sistema:

```rust
// Recebe primeira observação do Unity
let observations = unity.get_observations();

// Detecta automaticamente os sensores
let obs_spec = ObservationSpec::detect_from_observations(&observations);

// Imprime informações
obs_spec.print_info();
```

### 2. Informação Exibida

**Caso 1: Apenas Vector Observations**
```
╔════════════════════════════════════════════════════════════╗
║        📊 OBSERVATION SPECIFICATION DETECTED            ║
╚════════════════════════════════════════════════════════════╝

✅ Vector Observations:
   └─ Size: 62 dimensions

⚠️  No RayPerception sensors detected
   └─ Training with vector observations only

📏 Total Observation Size: 62 dimensions

💡 Model will be configured for:
   ✓ Single vector observation input
════════════════════════════════════════════════════════════
```

**Caso 2: Vector + RayPerception**
```
╔════════════════════════════════════════════════════════════╗
║        📊 OBSERVATION SPECIFICATION DETECTED            ║
╚════════════════════════════════════════════════════════════╝

✅ Vector Observations:
   └─ Size: 62 dimensions

✅ RayPerception Sensors Detected:
   Sensor 0:
   └─ Name: RayPerceptionSensor0
   └─ Estimated rays: ~20
   └─ Data per ray: ~5
   └─ Total size: 100

📏 Total Observation Size: 162 dimensions

💡 Model will be configured for:
   ✓ Multiple observation inputs (vector + ray perception)
════════════════════════════════════════════════════════════
```

## 🔧 Estrutura Detectada

### ObservationSpec

```rust
pub struct ObservationSpec {
    pub has_vector_obs: bool,        // Tem observações vetoriais?
    pub vector_obs_size: usize,      // Tamanho das obs vetoriais
    pub has_ray_perception: bool,    // Tem RayPerception?
    pub ray_perception_specs: Vec<RayPerceptionSpec>,
    pub total_obs_size: usize,       // Tamanho total
}
```

### RayPerceptionSpec

```rust
pub struct RayPerceptionSpec {
    pub name: String,           // Nome do sensor
    pub num_rays: usize,        // Número estimado de raios
    pub data_per_ray: usize,    // Dados por raio
    pub total_size: usize,      // Tamanho total
}
```

## 📊 Configuração Automática do Modelo

### Apenas Vector Obs

```rust
// obs_dim = 62
let model = ActorNetwork::new(62, action_dim, hidden_dim);

// ONNX gerado:
// Input: vector_observation [batch, 62]
```

### Vector + RayPerception

```rust
// obs_dim = 162 (62 + 100)
let model = ActorNetwork::new(162, action_dim, hidden_dim);

// ONNX gerado:
// Input 0: obs_0 [batch, 62]          <- Vector obs
// Input 1: obs_1 [batch, 100]         <- RayPerception
// Ou input único: vector_observation [batch, 162]
```

## 🎮 No Unity

### Configuração A: Vector Only

```csharp
public class MyAgent : Agent
{
    public override void CollectObservations(VectorSensor sensor)
    {
        // 62 observações
        sensor.AddObservation(transform.position);      // 3
        sensor.AddObservation(transform.rotation);      // 4
        sensor.AddObservation(rigidbody.velocity);      // 3
        // ... total 62
    }
}
```

**Detectado:**
```
✅ Vector Observations: 62 dimensions
⚠️  No RayPerception sensors
📏 Total: 62 dimensions
```

### Configuração B: Vector + Ray

```csharp
public class MyAgent : Agent
{
    // Ray Perception Sensor Component no Inspector:
    // - Rays Per Direction: 10
    // - Max Ray Degrees: 70
    // - Detectable Tags: 3
    
    public override void CollectObservations(VectorSensor sensor)
    {
        // 62 observações
        sensor.AddObservation(transform.position);      // 3
        // ... total 62
    }
    
    // RayPerception adiciona automaticamente ~100 obs
}
```

**Detectado:**
```
✅ Vector Observations: 62 dimensions
✅ RayPerception Sensors: 1 sensor, ~100 dimensions
📏 Total: 162 dimensions
```

## 🚀 Workflow Automático

### 1. Iniciar Treinamento

```bash
cargo run --bin rust-mlagents train --config config.yaml
```

### 2. Sistema Detecta Automaticamente

```
🔄 Connecting to Unity...
✅ Connected!
🔍 Detecting observation configuration...

╔════════════════════════════════════════════════════════════╗
║        📊 OBSERVATION SPECIFICATION DETECTED            ║
╚════════════════════════════════════════════════════════════╝

✅ Vector Observations: 62 dimensions
✅ RayPerception Sensors Detected: 1 sensor
📏 Total Observation Size: 162 dimensions

💡 Model configured for multiple inputs
```

### 3. Treinar com Config Correta

```
🤖 Creating SAC model...
   └─ obs_dim: 162
   └─ action_dim: 2
   └─ hidden_dim: 256

🎯 Starting training...
```

### 4. Export ONNX Automático

```
✓ Checkpoint saved at step 10000
✓ ONNX exported: sac_step_10000.onnx
   └─ Configured for:
      • Vector observations: 62
      • RayPerception: 100
      • Total inputs: 162
```

## 💡 Vantagens

### ✅ Automático
- Sem configuração manual
- Detecta sensores automaticamente
- Informa claramente o que foi detectado

### ✅ Flexível
- Suporta apenas vector obs
- Suporta vector + ray perception
- Suporta múltiplos sensores

### ✅ Confiável
- Valida configuração a cada episódio
- Avisa se configuração mudar
- Previne erros de dimensão

## 🔍 Verificação Durante Treinamento

O sistema verifica continuamente:

```rust
// A cada nova observação
if !obs_spec.matches(&new_observations) {
    println!("⚠️  WARNING: Observation configuration changed!");
    println!("   Expected: {} dimensions", obs_spec.total_obs_size);
    println!("   Received: {} dimensions", new_total);
    println!("   Training may become unstable!");
}
```

## 🐛 Troubleshooting

### "Observation size mismatch"

**Causa:** Unity mudou configuração durante treinamento

**Solução:**
1. Pare o treinamento
2. Verifique configuração no Unity
3. Reinicie o treinamento

### "ONNX has wrong input shape"

**Causa:** ONNX foi gerado para configuração diferente

**Solução:**
1. Verifique dimensões no Unity Inspector
2. Regenere ONNX com configuração correta
3. Use modelo correspondente

### RayPerception não detectado

**Causa:** Sensor não está enviando dados

**Solução:**
1. Verifique se componente está ativo no Unity
2. Verifique se há tags detectáveis configuradas
3. Reinicie Unity e treinamento

## 📝 Exemplo Completo

### Unity Setup

```csharp
public class RobotAgent : Agent
{
    // Inspector:
    // - Behavior Parameters
    //   └─ Vector Observation Space: 10
    // - Ray Perception Sensor 3D
    //   └─ Rays Per Direction: 5
    //   └─ Detectable Tags: Player, Enemy, Wall
    
    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(transform.localPosition);  // 3
        sensor.AddObservation(rigidbody.velocity);       // 3
        sensor.AddObservation(hasItem ? 1f : 0f);        // 1
        sensor.AddObservation(health);                    // 1
        sensor.AddObservation(ammo);                      // 1
        sensor.AddObservation(targetDirection);           // 1
        // Total: 10
    }
}
```

### Detecção Automática

```
╔════════════════════════════════════════════════════════════╗
║        📊 OBSERVATION SPECIFICATION DETECTED            ║
╚════════════════════════════════════════════════════════════╝

✅ Vector Observations:
   └─ Size: 10 dimensions

✅ RayPerception Sensors Detected:
   Sensor 0:
   └─ Name: RayPerceptionSensor0
   └─ Estimated rays: ~11 (5 per direction + forward)
   └─ Data per ray: ~4 (3 tags + distance)
   └─ Total size: 44

📏 Total Observation Size: 54 dimensions

💡 Model will be configured for:
   ✓ Multiple observation inputs (vector + ray perception)
════════════════════════════════════════════════════════════
```

### Modelo Criado

```rust
// Automaticamente configurado com:
let trainer = SACTrainer::new(
    54,          // ← obs_dim (10 + 44)
    2,           // action_dim
    config,
    device
)?;
```

### ONNX Gerado

```
✓ ONNX exported: robot_agent.onnx
  Inputs:
    - obs_0: [batch, 10]   (vector observations)
    - obs_1: [batch, 44]   (ray perception)
  OR combined:
    - vector_observation: [batch, 54]
```

## 📚 Referências

- `observation_spec.rs` - Código de detecção
- `unity_env.rs` - Integração com Unity
- `ONNX_GENERATION_GUIDE.md` - Export de modelos

---

**Status:** ✅ IMPLEMENTADO
**Detecção:** Automática na inicialização
**Suporte:** Vector Obs + RayPerception
**Export:** ONNX com inputs corretos
