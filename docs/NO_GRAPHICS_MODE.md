# No Graphics Mode

## ⚠️ Importante: `no_graphics` é diferente de outros engine settings

### Como funciona no ML-Agents

No Unity ML-Agents original (Python), o parâmetro `no_graphics` é especial:

- **NÃO é enviado via Side Channel** (como width, height, time_scale)
- **É um argumento de linha de comando** passado ao executável Unity

### Por que isso acontece?

O `no_graphics` precisa ser configurado **antes** do Unity inicializar o sistema gráfico, então não pode ser mudado em runtime via Side Channel.

## Como usar no rust-mlagents

### Opção 1: Iniciar Unity manualmente com --no-graphics

```bash
# No terminal 1: Inicie o executável Unity com --no-graphics
./Build/YourGame.app/Contents/MacOS/YourGame --no-graphics

# No terminal 2: Execute o treinamento
cargo run --release --bin rust-mlagents-learn -- config.yaml
```

### Opção 2: Build headless do Unity

Ao fazer o build no Unity, selecione:
- **Platform:** Linux (ou Mac/Windows)
- **Target:** Server Build (headless)

Isso cria um build que **sempre roda sem gráficos**.

### Opção 3: Desabilitar câmera no código Unity

```csharp
public class DisableRendering : MonoBehaviour
{
    void Start()
    {
        // Desabilita todas as câmeras
        Camera.main.enabled = false;
        
        // Ou desabilita rendering completamente
        Camera.main.targetDisplay = 8;
    }
}
```

## YAML Config

No `config.yaml`, o campo `no_graphics` serve apenas como **documentação** e para futuras implementações onde spawnaremos o Unity automaticamente:

```yaml
engine_settings:
  width: 84
  height: 84
  quality_level: 0
  time_scale: 100.0
  target_frame_rate: -1
  no_graphics: true  # ⚠️ Não é enviado via side channel
```

## Parâmetros enviados via Side Channel

Estes **SÃO** enviados em runtime e funcionam:

✅ `width` - Largura da tela  
✅ `height` - Altura da tela  
✅ `quality_level` - Qualidade gráfica (0-5)  
✅ `time_scale` - Velocidade da simulação  
✅ `target_frame_rate` - FPS alvo (-1 = unlimited)  
✅ `capture_frame_rate` - FPS de captura  

❌ `no_graphics` - Precisa ser argumento de linha de comando

## Verificar se está funcionando

No Unity, você pode verificar se os parâmetros foram aplicados:

```csharp
void Start()
{
    Debug.Log($"Screen: {Screen.width}x{Screen.height}");
    Debug.Log($"Quality: {QualitySettings.GetQualityLevel()}");
    Debug.Log($"TimeScale: {Time.timeScale}");
    Debug.Log($"Target FPS: {Application.targetFrameRate}");
}
```

Se os valores não estiverem sendo aplicados, verifique:

1. ✅ `SideChannelRegistration.cs` está criado e anexado a um GameObject
2. ✅ Script Execution Order está definido como -100
3. ✅ As mensagens de side channel estão sendo recebidas (veja os logs)

## Roadmap

🔮 **Futuro:** Quando implementarmos o spawn automático de ambientes Unity, o `no_graphics` será passado como argumento de linha de comando:

```rust
// Futuro
let unity_process = Command::new(&env_path)
    .arg("--no-graphics")  // ← Aqui
    .arg(format!("--port={}", port))
    .spawn()?;
```

Por enquanto, use as opções 1, 2 ou 3 acima. ✅
