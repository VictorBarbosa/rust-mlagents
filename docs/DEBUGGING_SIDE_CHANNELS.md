# Debugging Side Channels

## Como verificar se os parâmetros estão sendo enviados

### 1. No Rust (Trainer)

Ao executar o treinamento, você verá:

```
📤 Environment Parameters:
  - currentLesson: Number(5)
  - maxHorizontal: Number(4.0)
  - steps: Number(1)

⚙️  Engine Settings:
  - Resolution: 84x84
  - Quality Level: 0
  - Time Scale: 100x
  - Target FPS: -1
  - Capture FPS: 0

🔄 Resetando ambiente...
  📤 Side channel total: 88 bytes  ← Confirma que dados foram serializados
```

**O que significa:**
- `88 bytes` indica que os dados foram serializados corretamente
- Aproximadamente 20 bytes para engine config + ~24 bytes por environment parameter

### 2. No Unity (Receptor)

Crie um script de debug para verificar se os valores chegaram:

```csharp
using UnityEngine;
using Unity.MLAgents;

public class SideChannelDebugger : MonoBehaviour
{
    void Start()
    {
        // Aguarda alguns frames para os side channels serem processados
        StartCoroutine(DebugAfterDelay());
    }
    
    System.Collections.IEnumerator DebugAfterDelay()
    {
        yield return new WaitForSeconds(0.5f);
        
        Debug.Log("=== SIDE CHANNEL DEBUG ===");
        
        // Environment Parameters
        Debug.Log($"currentLesson: {Academy.Instance.EnvironmentParameters.GetWithDefault(\"currentLesson\", -999f)}");
        Debug.Log($"maxHorizontal: {Academy.Instance.EnvironmentParameters.GetWithDefault(\"maxHorizontal\", -999f)}");
        Debug.Log($"steps: {Academy.Instance.EnvironmentParameters.GetWithDefault(\"steps\", -999f)}");
        
        // Engine Settings (aplicados automaticamente)
        Debug.Log($"Screen: {Screen.width}x{Screen.height}");
        Debug.Log($"Quality: {QualitySettings.GetQualityLevel()}");
        Debug.Log($"TimeScale: {Time.timeScale}");
        Debug.Log($"Target FPS: {Application.targetFrameRate}");
        
        Debug.Log("=========================");
    }
}
```

**Anexe este script a um GameObject** e execute.

### 3. Verificando o Script de Registro

Certifique-se de que `SideChannelRegistration.cs` existe e está correto:

```csharp
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.SideChannels;

public class SideChannelRegistration : MonoBehaviour
{
    void Awake()
    {
        Debug.Log("[SideChannel] Registrando canais...");
        
        // Environment Parameters Channel
        var envChannel = new EnvironmentParametersChannel();
        SideChannelManager.RegisterSideChannel(envChannel);
        Debug.Log("[SideChannel] ✓ EnvironmentParametersChannel registrado");
        
        // Engine Configuration Channel
        var engineChannel = new EngineConfigurationChannel();
        SideChannelManager.RegisterSideChannel(engineChannel);
        Debug.Log("[SideChannel] ✓ EngineConfigurationChannel registrado");
    }
}
```

**Importante:**
1. ✅ Script anexado a GameObject na cena
2. ✅ GameObject está ativo desde o início
3. ✅ **Script Execution Order:** -100 (Edit → Project Settings → Script Execution Order)

### 4. Ordem de Eventos

Os side channels são processados na seguinte ordem:

```
1. Rust: Handshake com Unity
2. Rust: Envia RESET com side_channel data
3. Unity: SideChannelRegistration.Awake() ← Registra canais
4. Unity: Processa side channel messages
5. Unity: Aplica Engine Settings
6. Unity: Atualiza Environment Parameters
7. Unity: Seu código pode ler os parâmetros
```

### 5. Problemas Comuns

#### ❌ Valores não mudam

**Sintoma:** Valores ficam nos defaults, não mudam
**Causa:** Side channels não registrados antes do reset
**Solução:** 
- Verificar Script Execution Order = -100
- Verificar que GameObject está ativo
- Verificar logs do Unity para erros

#### ❌ `GetWithDefault` retorna default

**Sintoma:** `GetWithDefault("currentLesson", 0)` sempre retorna 0
**Causa:** Nome do parâmetro diferente entre YAML e Unity
**Solução:**
```yaml
# YAML - usar exatamente o mesmo nome
environment_parameters:
  currentLesson: 5  # ← Mesmo nome
```
```csharp
// Unity - usar exatamente o mesmo nome
Academy.Instance.EnvironmentParameters.GetWithDefault("currentLesson", 0)
                                                       // ↑ Mesmo nome
```

#### ❌ TimeScale não muda

**Sintoma:** Jogo continua na velocidade normal
**Causa:** Engine config não aplicado
**Solução:**
- Verificar que `EngineConfigurationChannel` está registrado
- Verificar que side_channel tem dados (ver bytes no log Rust)

### 6. Hex Dump dos Dados (Avançado)

Se precisar debugar o protocolo em baixo nível:

```rust
// Em cli/src/main.rs, adicione após combinar side channels:
println!("  📤 Side channel hex dump:");
for (i, chunk) in combined_side_channel.chunks(16).enumerate() {
    print!("    {:04x}:  ", i * 16);
    for byte in chunk {
        print!("{:02x} ", byte);
    }
    println!();
}
```

**Formato esperado:**
```
Environment Parameters Channel UUID: 1e 89 4c 53 0f 81 ea 11 a9 d0 82 24 85 86 04 00
Engine Config Channel UUID:         2c 34 51 e9 7e 4f ea 11 b2 38 78 4f 43 87 d1 f7
```

### 7. Teste Rápido

Para verificar que tudo funciona:

**1. YAML simples:**
```yaml
environment_parameters:
  test: 42.0

engine_settings:
  time_scale: 10.0
```

**2. Unity:**
```csharp
void Start()
{
    float test = Academy.Instance.EnvironmentParameters.GetWithDefault("test", -1f);
    Debug.Log($"Test value: {test}"); // Deve mostrar: 42.0
    Debug.Log($"TimeScale: {Time.timeScale}"); // Deve mostrar: 10.0
}
```

**3. Execute:**
```bash
cargo run --release --bin rust-mlagents-learn -- config.yaml
```

Se ver `42.0` e `10.0`, tudo está funcionando! ✅

### 8. Logs Completos

Para ver todos os detalhes, rode com debug:

```bash
RUST_LOG=debug cargo run --release --bin rust-mlagents-learn -- config.yaml
```

Isso mostrará todas as mensagens trocadas entre Rust e Unity.
