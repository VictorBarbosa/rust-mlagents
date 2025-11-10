# Sumário: Engine Settings e Environment Parameters

## ✅ O que ESTÁ funcionando

### 1. Engine Settings (enviados via Side Channel)

Todos estes parâmetros SÃO enviados e aplicados automaticamente no Unity:

```yaml
engine_settings:
  width: 84                  # ✅ FUNCIONA
  height: 84                 # ✅ FUNCIONA
  quality_level: 0           # ✅ FUNCIONA
  time_scale: 100.0          # ✅ FUNCIONA
  target_frame_rate: -1      # ✅ FUNCIONA
  capture_frame_rate: 0      # ✅ FUNCIONA
```

**Como verificar:**
```csharp
Debug.Log($"Screen: {Screen.width}x{Screen.height}");
Debug.Log($"TimeScale: {Time.timeScale}");
```

### 2. Environment Parameters (enviados via Side Channel)

```yaml
environment_parameters:
  currentLesson: 5           # ✅ FUNCIONA
  maxHorizontal: 4.0         # ✅ FUNCIONA
  steps: 1                   # ✅ FUNCIONA
```

**Como verificar:**
```csharp
float lesson = Academy.Instance.EnvironmentParameters.GetWithDefault("currentLesson", 0);
Debug.Log($"Current Lesson: {lesson}"); // Deve mostrar 5
```

## ❌ O que NÃO está funcionando (e por quê)

### `no_graphics`

```yaml
engine_settings:
  no_graphics: true  # ❌ NÃO É ENVIADO via side channel
```

**Por quê?**

No Unity ML-Agents original (Python), `no_graphics` é um **argumento de linha de comando**, não um parâmetro de side channel. O Unity precisa saber ANTES de inicializar o sistema gráfico.

**Como resolver:**

#### Opção 1: Linha de comando
```bash
./Build/Game.app/Contents/MacOS/Game --no-graphics
```

#### Opção 2: Server Build
No Unity: Build Settings → ✅ Server Build

#### Opção 3: Código
```csharp
Camera.main.enabled = false;
```

## 📋 Checklist de Implementação

Para que os engine settings e environment parameters funcionem, você precisa:

### No Unity:

1. ✅ Criar `SideChannelRegistration.cs`:
```csharp
using UnityEngine;
using Unity.MLAgents.SideChannels;

public class SideChannelRegistration : MonoBehaviour
{
    void Awake()
    {
        var envChannel = new EnvironmentParametersChannel();
        SideChannelManager.RegisterSideChannel(envChannel);
        
        var engineChannel = new EngineConfigurationChannel();
        SideChannelManager.RegisterSideChannel(engineChannel);
    }
}
```

2. ✅ Anexar script a um GameObject na cena

3. ✅ Definir Script Execution Order:
   - Edit → Project Settings → Script Execution Order
   - Adicionar `SideChannelRegistration`
   - Definir ordem: **-100**

### No Rust (já está implementado):

✅ Serialização de side channels  
✅ Combinação de múltiplos channels  
✅ Envio durante reset  
✅ Parsing de configuração YAML  

## 🧪 Teste Rápido

### 1. YAML mínimo (teste.yaml):
```yaml
environment_parameters:
  test: 42.0

engine_settings:
  time_scale: 10.0
```

### 2. Unity script:
```csharp
void Start()
{
    float test = Academy.Instance.EnvironmentParameters.GetWithDefault("test", -1f);
    Debug.Log($"Test: {test}, TimeScale: {Time.timeScale}");
}
```

### 3. Execute:
```bash
cargo run --release --bin rust-mlagents-learn -- teste.yaml
```

### 4. Resultado esperado:
```
Unity Console: Test: 42, TimeScale: 10
```

Se aparecer isso, FUNCIONOU! ✅

## 🔍 Debug

### Ver o que está sendo enviado:

Ao rodar o treinamento, você verá:
```
⚙️  Engine Settings:
  - Resolution: 84x84
  - Quality Level: 0
  - Time Scale: 100x
  - Target FPS: -1
  - Capture FPS: 0

🔄 Resetando ambiente...
  📤 Side channel total: 88 bytes
```

**88 bytes significa:**
- ~44 bytes: Engine config (6 valores × ~7 bytes)
- ~44 bytes: Environment params (3 parâmetros × ~14 bytes cada)

### Hex dump (se necessário):

Adicione no código Rust:
```rust
for (i, chunk) in combined_side_channel.chunks(16).enumerate() {
    println!("{:04x}: {:02x?}", i*16, chunk);
}
```

## 📚 Documentação Completa

- [`ENGINE_SETTINGS_EXPLAINED.md`](./ENGINE_SETTINGS_EXPLAINED.md) - Detalhes de cada parâmetro
- [`DEBUGGING_SIDE_CHANNELS.md`](./DEBUGGING_SIDE_CHANNELS.md) - Como debugar problemas
- [`NO_GRAPHICS_MODE.md`](./NO_GRAPHICS_MODE.md) - Como usar no_graphics
- [`example_config_full.yaml`](../example_config_full.yaml) - Config completo comentado

## 🎯 Conclusão

### ✅ Implementado e Funcionando:
- Engine settings via side channel
- Environment parameters via side channel
- Serialização e envio automático
- Configuração via YAML

### ⚠️ Requer Ação Manual:
- `no_graphics` precisa ser argumento de linha de comando OU server build
- SideChannelRegistration.cs precisa ser criado no Unity
- Script Execution Order precisa ser configurado

### 📊 Performance Esperada:

Com as configurações corretas:
```yaml
engine_settings:
  time_scale: 100.0
  quality_level: 0
  width: 84
  height: 84
```

Você pode treinar **até 100x mais rápido** que tempo real! 🚀

---

**Se algo não estiver funcionando, consulte [`DEBUGGING_SIDE_CHANNELS.md`](./DEBUGGING_SIDE_CHANNELS.md)**
