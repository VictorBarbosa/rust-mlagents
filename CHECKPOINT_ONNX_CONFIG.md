# 🔄 Configuração de Checkpoints e Export ONNX

## ✨ Sistema Automático

O treinamento SAC agora usa configuração YAML para controlar quando salvar checkpoints e exportar ONNX!

## 🎯 Configuração no YAML

### Parâmetros Principais

```yaml
# Frequência de checkpoints (em steps)
checkpoint_interval: 10000  

# Exportar ONNX automaticamente?
save_onnx: true
```

## 📊 Como Funciona

### Durante o Treinamento

O código verifica **automaticamente** a cada step:

```rust
if self.sac.should_checkpoint() {  // Verifica config.checkpoint_interval
    // Salva checkpoint
    self.sac.save_checkpoint("checkpoints/sac_step_10000.pt")?;
    
    // Se config.save_onnx = true
    if self.sac.config.save_onnx {
        self.sac.export_onnx("checkpoints/sac_step_10000")?;
    }
}
```

### Arquivos Gerados

**A cada checkpoint (ex: step 10000):**
- ✅ `checkpoints/sac_step_10000.pt` - Pesos do modelo
- ✅ `checkpoints/metadata.json` - Metadados (obs_dim, action_dim, config)
- ✅ `checkpoints/sac_step_10000.onnx` - ONNX (se `save_onnx: true`)

**No final do treinamento:**
- ✅ `checkpoints/sac_final.pt`
- ✅ `checkpoints/metadata.json`
- ✅ `checkpoints/sac_final.onnx` (se `save_onnx: true`)

## 🚀 Exemplos de Configuração

### 1. Desenvolvimento/Debug (checkpoints frequentes)

```yaml
checkpoint_interval: 1000   # A cada 1k steps
save_onnx: false            # Não exportar ONNX (mais rápido)
```

**Resultado:** Checkpoints rápidos sem overhead de ONNX export

### 2. Treinamento Normal (recomendado)

```yaml
checkpoint_interval: 10000  # A cada 10k steps
save_onnx: true             # Exportar ONNX também
```

**Resultado:** Checkpoints + ONNX prontos para Unity a cada 10k steps

### 3. Treinamento Longo

```yaml
checkpoint_interval: 50000  # A cada 50k steps
save_onnx: true
```

**Resultado:** Checkpoints menos frequentes, economiza espaço

### 4. Apenas Checkpoint Final

```yaml
checkpoint_interval: 0      # Desabilita checkpoints intermediários
save_onnx: true             # ONNX só no final
```

**Resultado:** Só salva no final do treinamento

## 📝 Workflow Completo

### 1. Configurar YAML

```yaml
# config.yaml
checkpoint_interval: 10000
save_onnx: true
hidden_layers: [256, 256]
# ... outras configs
```

### 2. Treinar

```rust
// O código lê o YAML automaticamente
let config = SACConfig::from_yaml("config.yaml")?;
let mut trainer = SACTrainer::new(obs_dim, action_dim, config, device)?;

// Treinar - checkpoints automáticos!
for episode in 0..num_episodes {
    // ...
    // ✅ A cada 10k steps: salva .pt + .json + .onnx
}
// ✅ No final: salva checkpoint final + ONNX
```

### 3. ONNX no Unity

```bash
# Copiar qualquer checkpoint para Unity
cp checkpoints/sac_step_10000.onnx Unity/Assets/ML-Agents/Models/

# Ou usar o checkpoint final
cp checkpoints/sac_final.onnx Unity/Assets/ML-Agents/Models/
```

## 🔍 Verificar Configuração Atual

### Ver config carregada:

```rust
println!("Checkpoint interval: {}", trainer.config.checkpoint_interval);
println!("Save ONNX: {}", trainer.config.save_onnx);
```

### Ver metadados salvos:

```bash
cat checkpoints/metadata.json
```

Saída:
```json
{
  "step": 10000,
  "obs_dim": 62,
  "action_dim": 2,
  "config": {
    "checkpoint_interval": 10000,
    "save_onnx": true,
    ...
  }
}
```

## 💡 Dicas

### 1. Desenvolvimento Rápido
Durante desenvolvimento/testes:
- `checkpoint_interval: 1000` (frequente)
- `save_onnx: false` (mais rápido)

### 2. Treinamento de Produção
Para treinamento final:
- `checkpoint_interval: 10000` (moderado)
- `save_onnx: true` (pronto para Unity)

### 3. Economizar Espaço
Se disco está cheio:
- `checkpoint_interval: 50000` (menos frequente)
- `save_onnx: false` (só gera ONNX manual depois)

### 4. Gerar ONNX Manualmente Depois
Se `save_onnx: false` e quiser ONNX depois:

```bash
# Usar o script Python
python3 convert_checkpoint_to_onnx.py checkpoints/sac_step_10000.pt
# ✅ Lê metadata.json automaticamente
# ✅ Gera sac_step_10000.onnx
```

## 🐛 Troubleshooting

### Checkpoint não está sendo salvo

**Verifique:**
1. `checkpoint_interval` > 0 no YAML
2. Treinamento passou do número de steps
3. Permissões de escrita na pasta `checkpoints/`

**Debug:**
```rust
println!("Should checkpoint: {}", trainer.should_checkpoint());
println!("Current step: {}", trainer.step);
println!("Interval: {}", trainer.config.checkpoint_interval);
```

### ONNX não está sendo gerado

**Verifique:**
1. `save_onnx: true` no YAML
2. Checkpoint foi salvo com sucesso
3. Script Python de conversão existe

**Solução manual:**
```bash
python3 convert_checkpoint_to_onnx.py checkpoints/sac_step_10000.pt
```

### Muitos arquivos sendo gerados

**Solução:** Aumente `checkpoint_interval`
```yaml
checkpoint_interval: 50000  # Menos frequente
```

### Treinamento lento por causa do ONNX export

**Solução:** Desabilite temporariamente
```yaml
save_onnx: false  # Gere ONNX manual no final
```

## 📊 Monitoramento

### Logs Durante Treinamento

```
Step 9999/50000 | Actor: -0.5, Critic: 0.3, Alpha: 0.2
✓ Checkpoint saved at step 10000
✓ ONNX exported at step 10000
Step 10001/50000 | Actor: -0.4, Critic: 0.2, Alpha: 0.2
...
Step 19999/50000 | Actor: -0.3, Critic: 0.1, Alpha: 0.2
✓ Checkpoint saved at step 20000
✓ ONNX exported at step 20000
```

## ⚙️ Integração com Scripts

### Script de Treinamento

```rust
use rl_core::trainers::sac::{SACTrainer, SACConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Carregar config do YAML
    let config = SACConfig::from_yaml("config.yaml")?;
    
    println!("📋 Config loaded:");
    println!("  checkpoint_interval: {}", config.checkpoint_interval);
    println!("  save_onnx: {}", config.save_onnx);
    
    // Criar trainer
    let mut trainer = SACTrainer::new(62, 2, config, Device::Cpu)?;
    
    // Treinar
    // ✅ Checkpoints automáticos conforme configurado
    
    Ok(())
}
```

## 📚 Referências

- `config_example.yaml` - Exemplo completo de configuração
- `AUTO_ONNX_EXPORT.md` - Detalhes do sistema de export automático
- `FINAL_SOLUTION.md` - Solução do formato ONNX que funciona

---

**Status:** ✅ IMPLEMENTADO
**Controle:** Via YAML config
**Export:** Automático conforme configuração
