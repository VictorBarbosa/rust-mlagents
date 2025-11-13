# 🎯 Problema Real - RayPerception Warning

## 🔍 Diagnóstico

Analisei o checkpoint atual e identifiquei o problema real:

```json
// metadata.json
{
  "obs_dim": 62,  // ← Modelo foi treinado COM APENAS 62 observações!
  "action_dim": 2
}
```

**Unity atual:** Vector (62) + RayPerceptionSensor (~100) = **2 sensores**  
**Modelo treinado:** Vector (62) apenas = **1 sensor**

## ⚠️ Por Que o Warning Aparece

```
The model does not contain an observation placeholder input 
for sensor component0 (rayperceptionsensor)
```

**Traduzindo:**  
"O modelo ONNX não tem input para o RayPerceptionSensor que você adicionou no Unity"

## ✅ Solução Definitiva

### Opção 1: Remover RayPerception do Unity (Temporário) ⚡

**Se você NÃO precisa do RayPerception AGORA:**

1. Unity → Selecionar Agent GameObject
2. Inspector → Ray Perception Sensor 3D component
3. Menu (⋮) → Remove Component
4. Salvar cena

**Resultado:**
- ✅ Warning desaparece
- ✅ Modelo funciona perfeitamente
- ✅ Sem retreinamento necessário

---

### Opção 2: RETREINAR Modelo COM RayPerception ⭐ (Recomendado)

**Se você PRECISA do RayPerception:**

#### Passo 1: Confirmar Configuração do Unity

Verifique que seu Agent tem:

```csharp
public class SimpleAgent : Agent
{
    public override void CollectObservations(VectorSensor sensor)
    {
        // Exatamente 62 float observations
        sensor.AddObservation(transform.position.x);     // 1
        sensor.AddObservation(transform.position.y);     // 1
        sensor.AddObservation(transform.localPosition.x); // 1
        sensor.AddObservation(transform.localPosition.y); // 1
        // ... total deve dar 62
    }
}
```

**E** tem o componente:
- **RayPerceptionSensor3D** (no Inspector)
  - Rays Per Direction: ? (ex: 10)
  - Detectable Tags: ? (ex: 3)
  - Obs por ray = tags + distância + hit = ~5-10
  - Total rays = (2 * rays_per_dir + 1) * obs_per_ray ≈ 100

#### Passo 2: Iniciar Treinamento DO ZERO

**IMPORTANTE:** Não use checkpoint antigo!

```bash
cd rust-mlagents

# Certifique-se que não vai carregar checkpoint antigo
rm -rf SimpleTrain/Assets/results/meu_treino/checkpoints/*.pt

# Treinar do zero
cargo run --bin rust-mlagents train --config config.yaml
```

#### Passo 3: Verificar Detecção

Quando o treino iniciar, você deve ver:

```
🔗 Unity connected: behavior 'SimpleAgent', 2 continuous actions
🔍 Waiting for first observation to detect sensor configuration...

╔════════════════════════════════════════════════════════╗
║   📊 OBSERVATION SPECIFICATION DETECTED            ║
╚════════════════════════════════════════════════════════╝

✅ Vector Observations: 62 dimensions
✅ RayPerception Sensor 0: 100 dimensions
📏 Total: 162 dimensions
```

**Se não aparecer o RayPerception:**
- Verifique que o componente está ativo no Unity
- Verifique que o treinamento conectou corretamente

#### Passo 4: Treinar até Convergência

```
Step 1000: Actor Loss: -0.2630, Critic Loss: 0.0153, Alpha: 0.2000
✓ Checkpoint saved at step 1000: .../SimpleAgent-1000.pt
✓ ONNX exported: .../SimpleAgent-1000.onnx
```

**Metadata agora terá:**
```json
{
  "obs_dim": 162,  // ← 62 + 100 !
  "action_dim": 2
}
```

#### Passo 5: Converter para ONNX com Múltiplos Inputs

Após treinamento, converter:

```bash
python3 fix_onnx_multi_input.py \
    SimpleTrain/Assets/results/meu_treino/checkpoints/SimpleAgent-10000.pt \
    --obs-sizes 62 100 \
    --output SimpleAgent_final.onnx
```

**Resultado:**
```
✅ ONNX exported successfully!

📊 Model configuration:
   Inputs:
      obs_0: [batch, 62]    ← Vector observations
      obs_1: [batch, 100]   ← RayPerception
   Outputs:
      continuous_actions: [batch, 2]
```

#### Passo 6: Usar no Unity

1. Copiar `SimpleAgent_final.onnx` para Unity Assets
2. Behavior Parameters → Model → Arraste o .onnx
3. ✅ **SEM WARNING!**

---

## 🧪 Como Verificar o Que Você Tem

### Verificar Metadata do Checkpoint

```bash
cat SimpleTrain/Assets/results/meu_treino/checkpoints/metadata.json | grep obs_dim
```

**Se mostrar:**
- `"obs_dim": 62` → Modelo SEM RayPerception
- `"obs_dim": 162` (ou >100) → Modelo COM RayPerception

### Verificar ONNX

```bash
python3 << 'EOF'
import onnx
m = onnx.load("SimpleAgent-1000.onnx")
print("Inputs:")
for i in m.graph.input:
    print(f"  {i.name}: {[d.dim_value for d in i.type.tensor_type.shape.dim]}")
EOF
```

**Esperado COM RayPerception:**
```
Inputs:
  obs_0: [1, 62]
  obs_1: [1, 100]
```

**Atual (SEM RayPerception):**
```
Inputs:
  obs_0: [1, 62]    ← Apenas vector!
```

### Verificar Unity

No console do Unity, quando conectar ao treinamento:

```
Connected to training server
Sending observations:
  [0] 62 floats (vector)
  [1] 100 floats (ray perception sensor)
```

Se mostrar apenas `[0]`, o RayPerception não está ativo!

---

## 📊 Comparação

| Aspecto | Modelo Atual | Modelo Correto |
|---------|-------------|---------------|
| **obs_dim** | 62 | 162 |
| **Sensores Unity** | Vector only | Vector + Ray |
| **Inputs ONNX** | 1 (obs_0) | 2 (obs_0, obs_1) |
| **Warning** | ❌ Sim | ✅ Não |
| **Funciona** | ⚠️ Parcial | ✅ Total |

---

## 🎯 Recomendação

### Se Não Precisa de RayPerception AGORA:
👉 **Use Opção 1** (Remover componente) - 5 minutos

### Se Precisa de RayPerception:
👉 **Use Opção 2** (Retreinar) - ~30 min a 2h dependendo do ambiente

---

## 🔧 Scripts Criados

### `fix_onnx_multi_input.py`

Converte checkpoint treinado (com flatten) para ONNX com múltiplos inputs:

```bash
python3 fix_onnx_multi_input.py checkpoint.pt --obs-sizes 62 100
```

**⚠️ IMPORTANTE:**  
- Só funciona se `obs_dim` no metadata = soma dos `obs-sizes`
- Se metadata tem `obs_dim: 62`, não pode converter para 162!
- **Precisa retreinar com 162 primeiro!**

---

## ❓ FAQ

### Q: Por que não posso usar o modelo atual com RayPerception?

**A:** O modelo foi treinado com apenas 62 inputs, mas Unity está enviando 162 (62+100). As dimensões não batem!

### Q: Quanto tempo leva para retreinar?

**A:** Depende:
- CartPole simples: 10-30 minutos
- Ambiente complexo: 1-3 horas
- Use GPU para acelerar

### Q: Posso "adicionar" as 100 dimensões no modelo existente?

**A:** Tecnicamente possível, mas não recomendado:
- Pesos treinados não sabem usar essas novas observações
- Melhor treinar do zero para aprender a usar RayPerception

### Q: O modelo vai melhorar com RayPerception?

**A:** **SIM!** RayPerception dá ao agente "visão" do ambiente:
- Detecta obstáculos
- Mede distâncias
- Identifica objetos
- Performance geralmente melhora 20-50%

### Q: Posso testar sem retreinar?

**A:** Sim! Use **Opção 1** (remover sensor) para testar se o modelo base funciona bem.

---

## 🎓 Lições Aprendidas

1. **Configuração Unity deve bater com Modelo treinado**
   - Não adicione sensores depois do treino!
   - Configure ANTES de treinar

2. **Metadata é a fonte da verdade**
   - `obs_dim` mostra o que foi treinado
   - Sempre verifique antes de usar modelo

3. **ONNX precisa ter inputs corretos**
   - 1 sensor → 1 input
   - N sensores → N inputs
   - Nomes: `obs_0`, `obs_1`, ...

4. **Retreinar é mais confiável que converter**
   - Conversão é "hack" temporário
   - Retreinamento aprende de verdade

---

## 📚 Próximos Passos

1. Decida: precisa de RayPerception?
   - **Não** → Opção 1 (5 min) ✅
   - **Sim** → Opção 2 (30 min-2h) ⭐

2. Se retreinar:
   - Configure Unity corretamente
   - Limpe checkpoints antigos
   - Treine do zero
   - Verifique metadata (obs_dim=162)
   - Converta para ONNX multi-input
   - Teste no Unity

3. Documente sua configuração:
   - Quantos sensores?
   - Quantas observações cada um?
   - Total de dimensões

---

**TL;DR:** Modelo atual tem 62 obs, Unity espera 162 (62+100). Solução: Remover RayPerception OU retreinar do zero com ele ativado.
