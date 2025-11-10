# Status: Multi-Environment Training

## 🚧 Estado Atual: Parcialmente Implementado

### ✅ O que funciona:

```yaml
env_settings:
  num_envs: 1  # ✅ Um ambiente funciona perfeitamente
```

### ❌ O que ainda não funciona:

```yaml
env_settings:
  num_envs: 2  # ❌ Múltiplos ambientes ignorados
```

## Por que `num_envs > 1` não funciona ainda?

### Arquitetura Atual

```
Rust Trainer (porta 5004)
    ↓
Unity Instance (porta 5004) ← Apenas 1 conexão
```

### Arquitetura Necessária para Multi-Env

```
Rust Trainer
    ↓
    ├─→ Unity Instance 1 (porta 5004)
    ├─→ Unity Instance 2 (porta 5005)
    ├─→ Unity Instance 3 (porta 5006)
    └─→ Unity Instance 4 (porta 5007)
```

## O que precisa ser implementado

### 1. Spawn de múltiplos processos Unity

```rust
// TODO: Implementar em cli/src/main.rs
pub struct UnityWorkerPool {
    workers: Vec<UnityWorker>,
}

struct UnityWorker {
    process: std::process::Child,
    grpc_server: GrpcServer,
    port: u16,
}

impl UnityWorkerPool {
    pub fn spawn(num_envs: usize, base_port: u16, env_path: &str) -> Result<Self> {
        let mut workers = Vec::new();
        
        for i in 0..num_envs {
            let port = base_port + i as u16;
            
            // Spawn Unity process
            let process = Command::new(env_path)
                .arg("--no-graphics")
                .arg(format!("--port={}", port))
                .spawn()?;
            
            // Create gRPC server for this worker
            let server = GrpcServer::new(port);
            
            workers.push(UnityWorker {
                process,
                grpc_server: server,
                port,
            });
        }
        
        Ok(Self { workers })
    }
}
```

### 2. Parallel data collection

```rust
// TODO: Coletar experiências de todos os workers em paralelo
pub async fn collect_rollouts(&mut self) -> Vec<Experience> {
    let mut experiences = Vec::new();
    
    // Collect from all workers in parallel
    let futures: Vec<_> = self.workers.iter_mut()
        .map(|worker| worker.collect_experience())
        .collect();
    
    let results = futures::future::join_all(futures).await;
    
    for result in results {
        experiences.extend(result?);
    }
    
    experiences
}
```

### 3. Broadcast de ações

```rust
// TODO: Enviar ações para todos os workers
pub async fn step_all(&mut self, actions: &ActionBatch) -> Vec<Observation> {
    let futures: Vec<_> = self.workers.iter_mut()
        .zip(actions.split())
        .map(|(worker, action)| worker.step(action))
        .collect();
    
    futures::future::join_all(futures).await
}
```

## Workaround Atual

Como `num_envs > 1` não funciona, use `num_areas` dentro do Unity:

### Solução 1: Multiple Training Areas (Recomendado)

```yaml
env_settings:
  num_envs: 1      # Apenas 1 Unity instance
  num_areas: 8     # 8 training areas dentro do Unity
```

**No Unity:**
```csharp
// Crie 8 áreas de treinamento na mesma cena
// Cada área tem seus próprios agentes independentes
```

**Vantagens:**
- ✅ Funciona agora
- ✅ Menos overhead (1 processo vs 8)
- ✅ Compartilha recursos gráficos

**Desvantagens:**
- ❌ Limitado por RAM de um processo
- ❌ Todos os agentes na mesma cena

### Solução 2: Time Scale Alto

```yaml
engine_settings:
  time_scale: 100.0  # 100x mais rápido
```

Em vez de 8 ambientes paralelos, rode 1 ambiente 100x mais rápido.

**Vantagens:**
- ✅ Simples
- ✅ Menos complexidade

**Desvantagens:**
- ❌ Menos diversidade de experiências

## Roadmap

### Milestone 1: Spawn básico ⏳
- [ ] Implementar UnityWorkerPool
- [ ] Spawn de N processos Unity
- [ ] Conexão de N servidores gRPC

### Milestone 2: Coleta paralela ⏳
- [ ] Collect rollouts em paralelo
- [ ] Combinar experiências de todos workers
- [ ] Balanceamento de carga

### Milestone 3: Otimizações 📅
- [ ] Worker reciclagem (evitar spawn/kill constante)
- [ ] Detecção de crash e restart
- [ ] Monitoramento de performance por worker

## Comparação com Python ML-Agents

### Python (funciona):
```python
env = UnityEnvironment(
    file_name=env_path,
    num_envs=8,  # Spawns 8 Unity processes
)
```

### Rust (planejado):
```rust
let pool = UnityWorkerPool::spawn(
    8,  // num_envs
    5004,  // base_port
    &env_path,
)?;
```

## Como Testar Quando Implementado

### Teste 1: Verificar spawn
```bash
# Deve mostrar 8 processos Unity
ps aux | grep Unity
```

### Teste 2: Verificar portas
```bash
# Deve mostrar portas 5004-5011 em uso
netstat -an | grep LISTEN | grep 500
```

### Teste 3: Verificar coleta paralela
```
[Step 100] Workers: 8 | Experiences/sec: 800
```

## Estimativa de Ganho

Com `num_envs: 8`:

- **Coleta de dados:** 8x mais rápido
- **Training:** Mesmo tempo (centralizado)
- **Speedup total:** ~5-7x (considerando overhead)

Combinado com `time_scale: 100.0`:

- **Speedup total:** ~500-700x 🚀

## Como Contribuir

Se você quiser implementar multi-env:

1. **Fork** o repositório
2. **Implemente** UnityWorkerPool em `cli/src/worker_pool.rs`
3. **Teste** com múltiplos Unity builds
4. **Submit PR** com testes

## Alternativa Atual (Recomendada)

Enquanto multi-env não está implementado, use:

```yaml
env_settings:
  num_envs: 1
  num_areas: 8  # Múltiplas áreas no Unity

engine_settings:
  time_scale: 100.0  # Compensa parcialmente
```

**Resultado:** ~100x speedup (vs ~500x com multi-env)

Ainda assim muito rápido! ✅

## Status de Prioridade

🔴 **Baixa prioridade** porque:
- Workaround com `num_areas` funciona bem
- `time_scale: 100.0` já dá bom speedup
- Implementação é complexa
- Requer testes extensivos

Se você precisa de multi-env URGENTE, considere usar o Python ml-agents temporariamente.
