# Documentação rust-mlagents

## 🚀 Início Rápido

1. **[Quick Start Guide](../rust-mlagents/README.md)** - Comece aqui!
2. **[Example Config](../example_config_full.yaml)** - Configuração completa comentada

## 📚 Guias Principais

### Configuração

- **[Engine Settings Explained](./ENGINE_SETTINGS_EXPLAINED.md)** - Detalhes de resolution, time_scale, quality_level, etc.
- **[Summary: Engine & Environment Parameters](./SUMMARY_ENGINE_PARAMS.md)** - O que funciona e o que não funciona
- **[No Graphics Mode](./NO_GRAPHICS_MODE.md)** - Como rodar sem interface gráfica

### Debugging

- **[Debugging Side Channels](./DEBUGGING_SIDE_CHANNELS.md)** - Como verificar se os parâmetros estão sendo enviados
- **[Multi-Environment Status](./MULTI_ENV_STATUS.md)** - Por que `num_envs > 1` não funciona (ainda)

### Unity Setup

- **[Unity Side Channels Setup](./UNITY_SIDE_CHANNELS_SETUP.md)** - Como configurar o Unity para receber parâmetros

## 🎯 Por Caso de Uso

### "Quero treinar o mais rápido possível"

1. Leia: [Engine Settings Explained](./ENGINE_SETTINGS_EXPLAINED.md) → Seção "Máxima Velocidade"
2. Use config:
```yaml
engine_settings:
  width: 84
  height: 84
  quality_level: 0
  time_scale: 100.0
  target_frame_rate: -1
```

### "Parâmetros do YAML não estão funcionando"

1. Leia: [Debugging Side Channels](./DEBUGGING_SIDE_CHANNELS.md)
2. Verifique: [Unity Side Channels Setup](./UNITY_SIDE_CHANNELS_SETUP.md)
3. Se ainda não funcionar: [Summary](./SUMMARY_ENGINE_PARAMS.md) → Checklist

### "Quero rodar sem gráficos"

1. Leia: [No Graphics Mode](./NO_GRAPHICS_MODE.md)
2. Use: `./Build/Game --no-graphics`
3. **Importante:** `no_graphics: true` no YAML NÃO funciona (veja por quê no doc)

### "Quero múltiplos ambientes paralelos"

1. Leia: [Multi-Environment Status](./MULTI_ENV_STATUS.md)
2. Use workaround: `num_areas: 8` no Unity
3. Aguarde implementação de `num_envs > 1`

## 🔍 Troubleshooting Rápido

### Problema: TimeScale não acelera

**Solução:**
```csharp
// No Unity
void Awake() {
    QualitySettings.vSyncCount = 0;
}
```

### Problema: Environment parameters retornam default

**Solução:**
1. Verificar `SideChannelRegistration.cs` existe
2. Verificar Script Execution Order = -100
3. Verificar nomes EXATOS entre YAML e Unity

### Problema: Resolução não muda

**Causa:** Unity Editor ignora `Screen.SetResolution`

**Solução:** Testar em build, não no editor

### Problema: `num_envs: 8` ignorado

**Causa:** Multi-env não implementado ainda

**Solução:** Use `num_areas: 8` no Unity

## 📖 Referência Completa

### YAML Config

```yaml
behaviors:
  AgentName:
    trainer_type: ppo
    hyperparameters:
      batch_size: 64
      buffer_size: 2048
      learning_rate: 0.0003
      beta: 0.005
      epsilon: 0.2
      lambd: 0.95
      num_epoch: 3
    network_settings:
      hidden_units: 64
      num_layers: 2
    reward_signals:
      extrinsic:
        gamma: 0.99
    max_steps: 5000
    summary_freq: 100

environment_parameters:
  param1: 1.0
  param2: 2.0

engine_settings:
  width: 84
  height: 84
  quality_level: 0
  time_scale: 100.0
  target_frame_rate: -1

env_settings:
  base_port: 5004
  num_envs: 1
  num_areas: 1
```

Ver [example_config_full.yaml](../example_config_full.yaml) para versão comentada.

### CLI Usage

```bash
# Treinamento básico
cargo run --release --bin rust-mlagents-learn -- config.yaml

# Com porta customizada
cargo run --release --bin rust-mlagents-learn -- config.yaml --base-port 6000

# Com device específico (cuda/cpu/metal)
cargo run --release --bin rust-mlagents-learn -- config.yaml --device cuda

# Ver ajuda
cargo run --release --bin rust-mlagents-learn -- --help
```

## 🏗️ Arquitetura

### Fluxo de Dados

```
1. YAML Config → Rust Parser
2. Rust → Serializa side_channel
3. Rust → gRPC → Unity
4. Unity → SideChannelManager → Aplica settings
5. Unity → Academy.EnvironmentParameters
```

### Side Channel Protocol

```
[UUID: 16 bytes][Length: 4 bytes][Data: N bytes]
```

**UUIDs:**
- Environment Params: `534c891e-810f-11ea-a9d0-822485860400`
- Engine Config: `e951342c-4f7e-11ea-b238-784f4387d1f7`

## 🔗 Links Úteis

- [Unity ML-Agents GitHub](https://github.com/Unity-Technologies/ml-agents)
- [Unity ML-Agents Docs](https://unity-technologies.github.io/ml-agents/)
- [Python API Reference](https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Python-LLAPI.md)
- [Side Channel Protocol](https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Python-LLAPI.md#communicating-additional-information-with-the-environment)

## 🤝 Contribuindo

Quer ajudar a implementar features faltantes?

1. **Multi-Environment Training** - [MULTI_ENV_STATUS.md](./MULTI_ENV_STATUS.md)
2. **Checkpoint Save/Load**
3. **Discrete Actions Support**
4. **Curiosity/GAIL Reward Signals**

Abra uma issue ou PR no GitHub!

## 📝 Changelog

Ver [CHANGELOG.md](./CHANGELOG.md) para histórico completo de mudanças.

## ❓ FAQ

### Q: Por que `no_graphics` não funciona?
**A:** É argumento de linha de comando, não side channel. [Veja aqui](./NO_GRAPHICS_MODE.md)

### Q: Por que `num_envs > 1` não funciona?
**A:** Não implementado ainda. [Veja status aqui](./MULTI_ENV_STATUS.md)

### Q: Como acelerar o treinamento?
**A:** Use `time_scale: 100.0` + baixa resolução. [Veja guia](./ENGINE_SETTINGS_EXPLAINED.md)

### Q: Como debugar side channels?
**A:** [Guia completo aqui](./DEBUGGING_SIDE_CHANNELS.md)

---

**Última atualização:** 2024-01-09  
**Versão:** 0.1.0  
**Compatibilidade:** Unity ML-Agents 2.0+
