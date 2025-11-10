# Engine Settings Explicados

## O que são Engine Settings?

Engine Settings são parâmetros que controlam como o Unity renderiza e executa a simulação. Eles são enviados via **Side Channel** em runtime.

## ✅ Parâmetros ENVIADOS via Side Channel

Estes parâmetros SÃO enviados do Rust para o Unity e aplicados automaticamente:

| Parâmetro | Tipo | Descrição | Valores Comuns |
|-----------|------|-----------|----------------|
| `width` | u32 | Largura da janela/tela | 84, 250, 800, 1920 |
| `height` | u32 | Altura da janela/tela | 84, 250, 600, 1080 |
| `quality_level` | i32 | Qualidade gráfica (0-5) | 0 (lowest), 5 (highest) |
| `time_scale` | f32 | Velocidade da simulação | 1.0 (normal), 20.0, 100.0 |
| `target_frame_rate` | i32 | FPS alvo | -1 (unlimited), 60, 30 |
| `capture_frame_rate` | i32 | FPS de captura | 0 (disabled), 60 |

### Exemplos de Uso

#### Treinamento Rápido (Máxima Performance)

```yaml
engine_settings:
  width: 84              # Baixa resolução
  height: 84
  quality_level: 0       # Qualidade mínima
  time_scale: 100.0      # 100x mais rápido
  target_frame_rate: -1  # Sem limite de FPS
```

**Resultado:** Treinamento ~100x mais rápido que tempo real

#### Visualização (Qualidade)

```yaml
engine_settings:
  width: 1280
  height: 720
  quality_level: 5       # Qualidade máxima
  time_scale: 1.0        # Velocidade normal
  target_frame_rate: 60  # 60 FPS
```

**Resultado:** Visual bonito para demonstrações

#### Inferência/Demo

```yaml
engine_settings:
  width: 800
  height: 600
  quality_level: 3       # Qualidade média
  time_scale: 1.0        # Velocidade normal
  target_frame_rate: 60
```

## ❌ Parâmetro NÃO enviado via Side Channel

### `no_graphics`

**Por que é especial?**

O Unity precisa saber se deve inicializar o sistema gráfico **antes** de começar. Isso não pode ser mudado em runtime.

**Como usar:**

### Opção 1: Argumento de linha de comando

```bash
# Mac
./Build/Game.app/Contents/MacOS/Game --no-graphics

# Linux
./Build/Game.x86_64 --no-graphics

# Windows
Build\Game.exe -batchmode -nographics
```

### Opção 2: Server Build

No Unity Editor:
1. File → Build Settings
2. Selecione Platform (Linux/Mac/Windows)
3. ✅ Marque "Server Build"
4. Build

Builds de servidor sempre rodam sem gráficos.

### Opção 3: Código Unity

```csharp
void Awake()
{
    #if !UNITY_EDITOR
    // Desabilita rendering fora do editor
    Camera.main.enabled = false;
    #endif
}
```

## Como Funciona Internamente

### 1. Serialização (Rust)

```rust
// rl_core/src/side_channel.rs
pub fn serialize_engine_config(config: &EngineConfig) -> Vec<u8> {
    // UUID do EngineConfigurationChannel
    // e951342c-4f7e-11ea-b238-784f4387d1f7
    let mut data = ENGINE_CONFIG_UUID.to_vec();
    
    // Serializa valores em little-endian
    data.extend(config.width.to_le_bytes());
    data.extend(config.height.to_le_bytes());
    data.extend(config.quality_level.to_le_bytes());
    data.extend(config.time_scale.to_le_bytes());
    data.extend(config.target_frame_rate.to_le_bytes());
    data.extend(config.capture_frame_rate.to_le_bytes());
    
    data
}
```

### 2. Envio (Rust)

```rust
// Durante o primeiro reset
let engine_data = serialize_engine_config(&config);
let env_params_data = serialize_environment_parameters(&params);
let combined = combine_side_channels(&[engine_data, env_params_data]);

server.reset_with_side_channel(combined).await?;
```

### 3. Recepção (Unity C#)

```csharp
// Unity ML-Agents package - automático
public class EngineConfigurationChannel : SideChannel
{
    public EngineConfigurationChannel()
    {
        ChannelId = new Guid("e951342c-4f7e-11ea-b238-784f4387d1f7");
    }
    
    protected override void OnMessageReceived(IncomingMessage msg)
    {
        var width = msg.ReadInt32();
        var height = msg.ReadInt32();
        var qualityLevel = msg.ReadInt32();
        var timeScale = msg.ReadFloat32();
        var targetFrameRate = msg.ReadInt32();
        var captureFrameRate = msg.ReadInt32();
        
        // Aplica configurações
        Screen.SetResolution(width, height, false);
        QualitySettings.SetQualityLevel(qualityLevel);
        Time.timeScale = timeScale;
        Application.targetFrameRate = targetFrameRate;
    }
}
```

### 4. Registro (Unity - Você precisa fazer isso)

```csharp
// SideChannelRegistration.cs
public class SideChannelRegistration : MonoBehaviour
{
    void Awake()
    {
        var channel = new EngineConfigurationChannel();
        SideChannelManager.RegisterSideChannel(channel);
    }
}
```

**⚠️ Script Execution Order: -100**

## Verificando se Funcionou

### No Rust

```
⚙️  Engine Settings:
  - Resolution: 84x84
  - Quality Level: 0
  - Time Scale: 100x
  - Target FPS: -1
  - Capture FPS: 0

🔄 Resetando ambiente...
  📤 Side channel total: 44 bytes  ← 44 bytes = engine config enviado
```

### No Unity

```csharp
void Start()
{
    Debug.Log($"Screen: {Screen.width}x{Screen.height}");
    Debug.Log($"Quality: {QualitySettings.GetQualityLevel()}");
    Debug.Log($"TimeScale: {Time.timeScale}");
    Debug.Log($"FPS Target: {Application.targetFrameRate}");
}
```

**Output esperado:**
```
Screen: 84x84
Quality: 0
TimeScale: 100
FPS Target: -1
```

Se os valores estiverem corretos, funcionou! ✅

## Troubleshooting

### Valores não mudam

**Problema:** Engine settings ficam nos defaults

**Causa:** EngineConfigurationChannel não registrado

**Solução:**
1. Criar `SideChannelRegistration.cs` (ver acima)
2. Anexar a GameObject na cena
3. Definir Script Execution Order = -100

### TimeScale não acelera

**Problema:** Simulação continua lenta mesmo com `time_scale: 100.0`

**Causas possíveis:**
1. VSync ativado (limita FPS)
   - Solução: `QualitySettings.vSyncCount = 0;`
2. FixedUpdate muito pesado
   - Solução: Otimizar física ou aumentar Fixed Timestep
3. Rendering muito pesado
   - Solução: Diminuir resolução e qualidade

### Resolução não muda

**Problema:** Tela continua no tamanho original

**Causa:** No Unity Editor, Screen.SetResolution é ignorado

**Solução:** Teste em build, não no Editor

## Performance Tips

### Máxima Velocidade

```yaml
engine_settings:
  width: 84
  height: 84
  quality_level: 0
  time_scale: 100.0
  target_frame_rate: -1
```

```csharp
void Awake()
{
    QualitySettings.vSyncCount = 0;
    Physics.autoSimulation = true;
    Time.fixedDeltaStep = 0.02f; // 50 Hz physics
}
```

**Resultado:** ~100x mais rápido

### Balanceado

```yaml
engine_settings:
  width: 250
  height: 250
  quality_level: 1
  time_scale: 20.0
  target_frame_rate: 60
```

**Resultado:** ~20x mais rápido, ainda visualizável

## Referências

- [Unity ML-Agents Documentation](https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Python-LLAPI.md)
- [EngineConfigurationChannel Source](https://github.com/Unity-Technologies/ml-agents/blob/main/com.unity.ml-agents/Runtime/SideChannels/EngineConfigurationChannel.cs)
- [Side Channel Protocol](https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Python-LLAPI.md#communicating-additional-information-with-the-environment)
